"""高考英语听力真题自动切分工具

用法（三选一）:
  1. 直接双击运行:  切分本文件(或exe)所在文件夹里的全部 MP3
  2. 把 MP3 拖到本程序图标上: 只切分拖入的文件
  3. 命令行: python split-listening-v4.py <文件或文件夹> [--no-pause]

输出: 每个 MP3 旁边生成 <原名>-切分结果/ 文件夹 (D01-D10.mp3 + timeline.json)
"""
import re, subprocess, sys, json, os, glob, shutil, time
import numpy as np

SR = 8000
MAX_PLAY_GAP = 30.0
MIN_PAIR_SPEECH = 6.0         # 一段双遍材料的最短时长(秒)。第一节最短的对话跨度约
                              # 8 秒，而提示音、题目指引这类假配对的跨度都不到 4 秒，
                              # 6 秒能把两者分开。按「跨度」而不是「语音块时长之和」
                              # 判：材料内部的小停顿会把后者拉低，害得短材料被误判
MARKER_SPECTRAL_MAX = 0.010   # 见 spectral_change 的说明与实测分布
N_SINGLE = 5                  # 第一节：播一遍的材料数
N_DOUBLE = 5                  # 第二节：播两遍的材料数（标准高考 5+5）
FF = None  # resolved ffmpeg binary

# ---------- ffmpeg resolution ----------

def resolve_ffmpeg():
    p = shutil.which("ffmpeg")
    if p:
        return p
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None

# ---------- audio decode ----------

def decode_all(path):
    r = subprocess.run([FF, "-hide_banner", "-loglevel", "error", "-i", path,
                        "-f", "s16le", "-ac", "1", "-ar", str(SR), "pipe:1"],
                       capture_output=True)
    if r.returncode != 0 or not r.stdout:
        raise RuntimeError("无法读取音频文件: " + os.path.basename(path))
    return np.frombuffer(r.stdout, dtype=np.int16).astype(np.float32) / 32768.0

def seg(x, start, end):
    return x[int(start * SR):int(end * SR)]

# ---------- silence / chunks ----------

def detect_silences_np(x, sr=SR, noise_db=-35.0, min_sil=0.6):
    """静音检测：逐样本幅度低于 noise_db 且持续 >= min_sil 的区间。

    注：本文件曾尝试改用「短时 RMS 包络 + 相对阈值 + 迟滞」，并用 sweep.py 对
    36 组参数做网格扫参，结果最好也只有 6/8 达标（且放弃了临沂、无锡两份，
    也没救回潍坊），不如当前这版（7/8）。所以保留原做法。两遍切分不一致的
    问题改由 align_pair() 的整段波形对齐从构造上消除，不再靠调静音阈值。
    """
    thresh = 10.0 ** (noise_db / 20.0)
    sil = np.abs(x) < thresh
    d = np.diff(sil.astype(np.int8))
    starts = np.flatnonzero(d == 1) + 1
    ends = np.flatnonzero(d == -1) + 1
    if len(sil) and sil[0]:
        starts = np.r_[0, starts]
    if len(sil) and sil[-1]:
        ends = np.r_[ends, len(sil)]
    need = int(min_sil * sr)
    return [(s / sr, e / sr) for s, e in zip(starts, ends) if e - s >= need]


def detect_silences(x, sr=SR, min_sil=0.6):
    """静音检测的自适应入口：默认 −35dB，只在「切得太粗」时放宽门限重试。

    底噪偏高的录音（教室现场录的、电话/网络转录的）在 −35dB 下几乎没有一段样本
    能落到门限以下，材料被整段粘成 43~76 秒的巨块，后面配对、切分全线失效。
    这里给一道安全网：语音块平均跨度超过 ADAPT_SPAN 秒就依次用 −31dB、−28dB 重来。

    触发条件定得很松，默认路径因此完全不变 —— 实测 102 份样本在 −35dB 下平均每
    块 7~14 秒（最粗的一份 84 块 / 1139 秒 = 每块 13.6 秒），没有一份够得着 25 秒。
    """
    dur = len(x) / sr
    tries = [("−35dB", detect_silences_np(x, sr, -35.0, min_sil))]
    if len(build_chunks(tries[0][1], dur)) * ADAPT_SPAN >= dur:
        return tries[0][1]
    for db in (-31.0, -28.0):
        sil = detect_silences_np(x, sr, db, min_sil)
        tries.append((f"−{abs(db):.0f}dB", sil))
        if len(build_chunks(sil, dur)) * ADAPT_SPAN >= dur:
            print(f"  · 这份录音底噪偏高，静音门限已从 −35dB 放宽到 −{abs(db):.0f}dB")
            return sil
    # 放宽到底仍不够细，取块数最多的那次（总比一整块强）
    best = max(tries, key=lambda t: len(build_chunks(t[1], dur)))
    print(f"  · 这份录音底噪偏高，静音门限已放宽到 {best[0]}，"
          f"但仍切不细，切分结果可能不准，请人工核对")
    return best[1]


def build_chunks(sil, dur):
    chunks, pos = [], 0.0
    for s, e in sil:
        if s > pos:
            chunks.append([pos, s])
        pos = e
    if pos < dur:
        chunks.append([pos, dur])
    return chunks

def gap_before(chunks, i):
    return chunks[i][0] - chunks[i - 1][1] if i > 0 else 999.0

def gap_after(chunks, i):
    return chunks[i + 1][0] - chunks[i][1] if i + 1 < len(chunks) else 999.0

# ---------- waveform correlation ----------

def xcorr_best(a, b, max_lag=SR // 4):
    n = min(len(a), len(b))
    if n < SR // 10:
        return -1.0
    a = np.asarray(a[:n], dtype=np.float64)
    b = np.asarray(b[:n], dtype=np.float64)
    a -= a.mean(); b -= b.mean()
    sa = np.concatenate([[0.0], np.cumsum(a * a)])
    sb = np.concatenate([[0.0], np.cumsum(b * b)])
    if sa[-1] < 1e-12 or sb[-1] < 1e-12:
        return -1.0
    N = 1 << (2 * n).bit_length()
    fa, fb = np.fft.rfft(a, N), np.fft.rfft(b, N)
    pos = np.fft.irfft(fa * np.conj(fb), N)
    neg = np.fft.irfft(fb * np.conj(fa), N)
    L = min(max_lag, n - SR // 10)
    best = -1.0
    if L >= 0:
        l = np.arange(0, L + 1)
        na2 = sa[n] - sa[l]
        nb2 = sb[n - l]
        ok = (na2 > 1e-12) & (nb2 > 1e-12) & (n - l >= SR // 10)
        if ok.any():
            best = max(best, float((pos[l[ok]] / np.sqrt(na2[ok] * nb2[ok])).max()))
        m = np.arange(1, L + 1)
        ok = n - m >= SR // 10
        if ok.any():
            na2 = sa[n - m]
            nb2 = sb[n] - sb[m]
            good = (na2 > 1e-12) & (nb2 > 1e-12)
            if good.any():
                best = max(best, float((neg[m[good]] / np.sqrt(na2[good] * nb2[good])).max()))
    return best

# ---------- marker (chime) discovery ----------

def spectral_change(x, s, e, n_fft=512, hop=256):
    """一段音频相邻帧之间频谱变化的平均值。

    提示音是固定乐音/和弦，各帧频谱几乎不变，这个值很小；
    人声即使只有一两个字，频谱也在不停变，这个值明显更大。
    用来防止「逐字重复的合成播报」混进提示音簇里。
    """
    a = x[int(s * SR):int(e * SR)]
    if len(a) < n_fft * 3:
        return None
    w = np.hanning(n_fft)
    frames = np.lib.stride_tricks.sliding_window_view(a, n_fft)[::hop] * w
    spec = np.abs(np.fft.rfft(frames, axis=1))
    spec = spec / (spec.max(axis=1, keepdims=True) + 1e-9)
    return float(np.median(np.abs(np.diff(spec, axis=0)).mean(axis=1)))


def find_markers(x, chunks, min_cluster=4):
    # 先用频谱平稳性把「人声短语」挡在外面，再做聚类。
    # 实测 4 份样本：60 个真提示音的频谱变化最大值 0.0035，179 个非提示音短语的
    # 最小值 0.0174，两者零重叠。少了这一步，一段反复出现的合成播报（如
    # 「请听下面一段对话」）就可能凭互相关聚类劫持整个提示音簇，第一节边界全崩。
    cand = [i for i, (a, b) in enumerate(chunks) if 0.6 <= b - a <= 2.5]
    cand = [i for i in cand
            if (spectral_change(x, *chunks[i]) or 1.0) <= MARKER_SPECTRAL_MAX]
    n = len(cand)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    for ai in range(n):
        for bi in range(ai + 1, n):
            ia, ib = cand[ai], cand[bi]
            if abs((chunks[ia][1] - chunks[ia][0]) - (chunks[ib][1] - chunks[ib][0])) > 0.30:
                continue
            if xcorr_best(seg(x, *chunks[ia]), seg(x, *chunks[ib])) > 0.90:
                union(ai, bi)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(cand[i])
    if not groups:
        return []
    # 所有够大的簇都要，不要只留最大的：一份录音里可能有好几族提示音（实测
    # 单元检测卷（5）第一节的提示音是 1.22 秒一族、第二节是 1.26 秒一族；只留最大的
    # 那一族，第一节那 5 声「叮」就漏了，而它们互相之间相关度 0.99、残差 0.27，会被
    # 拿去当配对证据，把两道不同的题缝成一段材料 —— 正是那份卷子 D01 只有 1.2 秒的
    # 根源）。
    # min_cluster 从 6 降到 4：第一节只有 5 段，提示音一族最多也就 5 声。放心的依据
    # 是上面那道频谱闸（60 个真提示音 ≤0.0035、179 个非提示音短语 ≥0.0174，零重叠），
    # 能聚成簇的本来就只会是提示音。
    out = []
    for g in groups.values():
        if len(g) >= min_cluster:
            out.extend(g)
    return sorted(out)

# ---------- repeat pair detection ----------

def close_dur(a, b, tol_abs=0.10, tol_rel=0.04):
    """两段语音的时长是否「基本一样」（同一段材料两遍的静音切分可能略有出入）"""
    return abs(a - b) <= max(tol_abs, tol_rel * max(a, b))


def match_run(durs, i, j, n):
    """匹配从 i / j 开始的两串语音块，返回一一对应的 (ci, cj) 索引对。

    允许 1:2 和 2:1 合并。原因：同一段内容在两遍里被静音切成不同块数是常态
    （一遍一整块，另一遍被切成两块），逐块比时长一断就整段漏掉 —— 湘潭卷就是
    这样丢了第 7 段：第一遍 #87 是 25.77 秒整块，第二遍 #90+#91 是 9.14+16.03
    =25.17 秒，逐块比对在第 2 项就断了。
    """
    out, p, q = [], i, j
    while p < j and q < n:
        a, b = durs[p], durs[q]
        if close_dur(a, b):
            out.append((p, q)); p += 1; q += 1
        elif p + 1 < j and close_dur(a + durs[p + 1], b):
            out.append((p, q)); p += 2; q += 1
        elif q + 1 < n and close_dur(a, b + durs[q + 1]):
            out.append((p, q)); p += 1; q += 2
        else:
            break
    return out


def chunk_pair_ok(x, chunks, ci, cj):
    """这两个语音块是不是同一段内容的两次播放？

    先走旧判据（单块互相关 >PAIR_CORR），过不了再走容错的：两遍的起点可能错开 1 秒
    以上 —— 静音检测把同一段内容切成长度不同的块时，「块起点之差」就不等于真实位移。
    实测 单元检测卷（5）的 234.70~265.14 与 267.53~296.37：真实位移 31.232 秒，块起点
    差 32.83 秒，差了 1.6 秒；xcorr_best 只搜 ±0.25 秒，算出 0.019 直接判死，而按真实
    位移比下来残差只有 0.12（0.5 秒窗口剖面除头一窗外全是 0.02~0.04）—— 是真配对。
    estimate_lag 容许 ±PAIR_LAG_TOL 秒的错位，而残差的可分性远好于裸相关度（真配对
    ≤0.12，无关片段 ≥1.0），所以拿它当第二条判据。两条是「或」的关系，只放宽不收紧。
    """
    if xcorr_best(seg(x, *chunks[ci]), seg(x, *chunks[cj])) > PAIR_CORR:
        return True
    _lag, r = estimate_lag(x, chunks[ci][0], chunks[ci][1],
                           chunks[cj][0], chunks[cj][1], PAIR_LAG_TOL)
    return r is not None and r <= CHUNK_RESID


def find_repeat_pairs(x, chunks, min_chunks=2, min_total=3.0, marker_set=()):
    n = len(chunks)
    durs = [b - a for a, b in chunks]

    cands = []
    for i in range(n):
        for j in range(i + 1, n):
            if i > 0 and j > 0 and close_dur(durs[i - 1], durs[j - 1]):
                continue
            pr = match_run(durs, i, j, n)
            if not pr:
                continue
            tot = sum(durs[c] for c, _ in pr)
            # single-chunk matches allowed only for long chunks (material with no
            # internal silence is one chunk per play)
            if tot >= min_total and (len(pr) >= min_chunks or (len(pr) == 1 and tot >= 8.0)):
                cands.append((i, j, pr))
    cands.sort(key=lambda m: -sum(durs[c] for c, _ in m[2]))
    used = set()
    confirmed, soft = [], []
    for i, j, pr in cands:
        if any(t in used for t in (c for pr_ in pr for c in pr_)):
            continue
        # 证据里的提示音必须剔除：提示音是同一段录音被反复播放，互相关天然接近 1.0。
        # 潍坊卷有一段假配对（主体 "#29 vs #35" 相关只有 0.015），靠两个提示音
        # （corr 0.956 / 0.999）蒙混通过了确认，反而把真正的 "#34 vs #35"(0.979)
        # 挤出榜外，导致整份卷少切一段。
        ev = [(durs[ci], ci, cj) for ci, cj in pr
              if ci not in marker_set and cj not in marker_set]
        if not ev:
            continue
        ev.sort(reverse=True)
        ok = chunk_pair_ok(x, chunks, ev[0][1], ev[0][2])
        if not ok:
            ok = sum(1 for _, ci, cj in ev[:5] if chunk_pair_ok(x, chunks, ci, cj)) >= 2
        if not ok:
            continue
        # 上面只是单块粗筛（便宜）。时长链提出的边界常常吞进了提示音/题目指引，
        # 单块相关度照样很高，必须再用整段对齐精筛一次，否则会切出一段
        # 「含提示音、少了半段材料」的垃圾。
        anc = anchor_pair(x, chunks, i, j, pr[-1][0] + 1, pr[-1][1] + 1, marker_set)
        if anc is None:
            # 时长链和单块证据都认，只有整段波形对不上：多半是这段被真人读了两遍
            # （实测 11-标速(美音2) 第 4 题，两遍时长都是 11.25 秒、位置紧邻，但整段
            # 残差 1.00 —— 两次朗读不可能采样级对齐）。这种段照切，只是打旗子让人
            # 听一下，不能悄悄丢掉（丢了整份卷就少一段）。
            soft.append(_trim_markers(chunks, i, pr[-1][0] + 1, j, pr[-1][1] + 1,
                                      marker_set))
            continue
        a, c, b, d = anc
        used.update(range(a, b))
        used.update(range(c, d))
        confirmed.append((a, c, b, d))
    out = confirmed + rescue_pairs(x, chunks, marker_set, confirmed)
    # 软配对最后加：只能填空，不许抢已配上的块（硬配对永远优先）
    used = set()
    for a, c, b, d in out:
        used.update(range(a, b))
        used.update(range(c, d))
    for a, c, b, d in soft:
        if a >= b or c >= d:
            continue
        if any(t in used for t in range(a, b)) or any(t in used for t in range(c, d)):
            continue
        used.update(range(a, b))
        used.update(range(c, d))
        out.append((a, c, b, d))
    return sorted(out)


def _trim_markers(chunks, i1, j1, i2, j2, marker_set):
    """把时长链给的区间两端的提示音块去掉（波形判据在这类段上不可用）。

    软配对（真人读两遍）整段残差恒在 1.0 附近，trim_bounds 那种「对不上就往里收」
    的判据会把材料一路剪到只剩一半。所以只做结构性修剪：两端的块只要被判成提示
    音就摘掉，其余一律不动。
    """
    while i1 < j1 - 1 and i1 in marker_set:
        i1 += 1
    while i2 < j2 - 1 and i2 in marker_set:
        i2 += 1
    while j1 > i1 + 1 and j1 - 1 in marker_set:
        j1 -= 1
    while j2 > i2 + 1 and j2 - 1 in marker_set:
        j2 -= 1
    return i1, j1, i2, j2


def rescue_pairs(x, chunks, marker_set, pairs, min_total=8.0):
    """主通道没配上的地方再找一遍，把漏掉的第二遍捞回来。

    主通道靠「语音块时长链 + 单块相关度 >PAIR_CORR」提配对：时长链要求两遍被静音
    切成差不多的块，可有些制品两遍的切法差得远（实测 7.14 秒的块对上 8.67 秒的
    块），整段就漏了；也有些两遍做过变速/剪辑，相关度只有 0.4~0.6。这里改用最直接
    的办法 —— 只看主通道没覆盖的语音块，两两比开头 2 秒的波形（便宜），像的再交给
    reanchor 定边界（reanchor 只认整段波形，不看时长链）。

    晚于主通道执行：只往结果里加、不抢已经配好的块，所以已有结果不会变。
    """
    used = set()
    for i1, j1, i2, j2 in pairs:
        used.update(range(i1, i2))
        used.update(range(j1, j2))
    durs = [b - a for a, b in chunks]
    free = [i for i in range(len(chunks)) if i not in used and i not in marker_set]
    out = []
    for i in free:
        if i in used:
            continue
        head = seg(x, chunks[i][0], chunks[i][0] + 2.0)
        for j in free:
            if j <= i or j in used:
                continue
            if chunks[j][0] - chunks[i][0] > 150.0:
                break
            if xcorr_best(head, seg(x, chunks[j][0], chunks[j][0] + 2.0),
                          max_lag=SR // 8) <= RESCUE_CORR:
                continue
            res = reanchor(x, chunks, i, j, i + 1, j + 1,
                           chunks[j][0] - chunks[i][0], marker_set, RESCUE_SPAN)
            if res is None:
                # 波形怎么对都对不齐（两遍被变速/剪辑，整块相关度只有 0.4~0.6）：
                # 退回时长链给的边界。但要求两遍总长接近，免得时长链一路凑到别的
                # 材料上，凑出一段又长又假的配对。
                pr = match_run(durs, i, j, len(chunks))
                if not pr:
                    continue
                b, d = pr[-1][0] + 1, pr[-1][1] + 1
                if not close_dur(chunks[b - 1][1] - chunks[i][0],
                                 chunks[d - 1][1] - chunks[j][0]):
                    continue
                # 时长链凑出来的边界仍要用波形过一遍：两道不同题的时长也可能碰巧
                # 接近（见 RESCUE_RESID 的实测）。只挡「完全不像」的，变速/剪辑过的
                # 两遍还能过。
                lag_r, _ = estimate_lag(x, chunks[i][0], chunks[b - 1][1],
                                        chunks[j][0], chunks[d - 1][1])
                rr = (None if lag_r is None else
                      span_resid_int(x, chunks[i][0], chunks[b - 1][1],
                                     round(lag_r * SR) / SR))
                if rr is None or rr > RESCUE_RESID:
                    continue
                res = (i, j, b, d)
            a, c, b, d = res
            if any(t in used for t in range(a, b)) or any(t in used for t in range(c, d)):
                continue
            if sum(durs[t] for t in range(a, b)) < min_total:
                continue
            used.update(range(a, b))
            used.update(range(c, d))
            out.append((a, c, b, d))
            break
    return out

def _match_runs(x, chunks, ci, cj):
    """find (m, k) such that chunks[ci:ci+m] and chunks[cj:cj+k] are the same
    recording (tolerates silence-split differences: 1 chunk vs several)"""
    for m, k in ((1, 1), (1, 2), (2, 1), (1, 3), (3, 1), (2, 2)):
        if ci + m > len(chunks) or cj + k > len(chunks):
            continue
        a0, a1 = chunks[ci][0], chunks[ci + m - 1][1]
        b0, b1 = chunks[cj][0], chunks[cj + k - 1][1]
        if abs((a1 - a0) - (b1 - b0)) > 1.5:
            continue
        if xcorr_best(seg(x, a0, a1), seg(x, b0, b1)) > 0.75:
            return m, k
    return None

def extend_pair_ends(x, chunks, marker_set, pairs, max_rounds=4):
    """extend each pair's boundaries while adjacent audio is the same recording;
    guards: no markers, gap <= 2s on both sides, p1 stays before p2"""
    out = []
    for i1, j1, i2, j2 in pairs:
        # forward
        for _ in range(max_rounds):
            if i2 >= j1 or j2 >= len(chunks) or i2 in marker_set or j2 in marker_set:
                break
            if (chunks[i2][0] - chunks[i2 - 1][1] > 2.0
                    or chunks[j2][0] - chunks[j2 - 1][1] > 2.0):
                break
            mk_ = _match_runs(x, chunks, i2, j2)
            if mk_ is None or i2 + mk_[0] > j1:
                break
            i2 += mk_[0]
            j2 += mk_[1]
        # backward
        for _ in range(max_rounds):
            if i1 <= 0 or j1 <= 0 or i1 - 1 in marker_set or j1 - 1 in marker_set:
                break
            if (chunks[i1][0] - chunks[i1 - 1][1] > 2.0
                    or chunks[j1][0] - chunks[j1 - 1][1] > 2.0):
                break
            mk_ = _match_runs(x, chunks, i1 - 1, j1 - 1)
            if mk_ is None or j1 - mk_[1] <= i1 - mk_[0]:
                break
            i1 -= mk_[0]
            j1 -= mk_[1]
        out.append((i1, j1, i2, j2))
    return out

def merge_pair_fragments(chunks, pairs):
    """merge fragment pairs of the same play-pair: same time shift (p2-p1) and
    adjacent on both sides (silence splitting may differ between the two plays)"""
    merged = []
    for p in sorted(pairs, key=lambda t: t[0]):
        i1, j1, i2, j2 = p
        if merged:
            m = merged[-1]
            shift_p = chunks[j1][0] - chunks[i1][0]
            shift_m = chunks[m[1]][0] - chunks[m[0]][0]
            if (i1 - m[2] <= 2 and j1 - m[3] <= 2 and i1 >= m[0] and j1 >= m[1]
                    and abs(shift_p - shift_m) < 2.0):
                m[2] = max(m[2], i2)
                m[3] = max(m[3], j2)
                continue
        merged.append([i1, j1, i2, j2])
    return [tuple(m) for m in merged]

# ---------- two-play alignment: 把「两遍一致」变成可测量的判据 ----------

ALIGN_SHIFT = 2.0    # 在 p2 附近搜索 lag 的最大范围(秒)
EDGE_WIN = 4.0       # 判定局部抬升时，两端各取多长的窗口(秒)
EDGE_CHECK = 1.8     # 两端残差 / 主体残差 超过此值 → 边界可疑
EDGE_MIN = 0.15      # 两端残差的绝对下限：比值是相对的，主体残差小到 0.01 时两端
                     # 哪怕只到 0.025（两遍耳朵完全听不出区别）比值也有 2.3，照样报。
                     # 实测 20 份高二听力里 11 条「两端多带了内容」全是这种：两端残差
                     # 0.024~0.078，而真正多切了东西的段两端残差是 0.31~1.50，中间空
                     # 着一档（次高 0.078），取 0.15 正好落在空档里
OUTLIER_RESID = 2.5  # 段残差 / 本卷残差中位数 超过此值 → 建议抽听
OUTLIER_MIN = 0.12   # 离群判定的绝对下限：光看「比本卷中位数高 2.5 倍」会误报 ——
                     # MP3 复制噪声让各段残差落在 0.003~0.05，中位数常常只有 0.006，
                     # 于是 0.05 的段也被判成离群，实测 20 份高二听力因此凭空多了 22 条
                     # 「96%~97% 明显低于 99%」的提醒（0.05 换算过来差 3 个百分点，
                     # 两遍耳朵完全听不出区别）。加一道绝对下限：残差本身不到 0.12
                     # （≈ RESID_OK 的三分之一）一律不提醒；真有问题的段残差是 0.25~0.3，
                     # 而且另有切点核对会先报出来
MAX_SINGLE = 26.0    # 单遍材料的最长时长(秒)，超过者为「第二节说明」之类的块
ADAPT_SPAN = 25.0    # 语音块平均跨度超过这么多秒 → 认定静音门限太严（底噪偏高），
                     # 见 detect_silences 的实测：正常录音每块 7~14 秒
MIN_SINGLE = 8.0     # 单遍材料的最短时长(秒)，短于此多半是切碎了（实测真材料最短 9.8 秒）
SPLIT_GAP = 3.0      # 无提示音兜底时，把「答题静音」认作分界的间隔(秒)
RESID_OK = 0.35      # 「这两遍确实是同一段」的整段残差上限（实测真配对 ≤0.31）
FINE_WIN = 2.0       # 亚样本位移精化用的短窗(秒)。实测这个位移是全段常量（同一段
                     # 材料前后各 12 秒搜出来的最优位移完全相同），2 秒和 24 秒的
                     # 结果一致（吻合度差 ≤0.1%），取短窗纯粹为了省时间
FINE_RANGE = 0.30    # 分数位移搜索范围(样本)
FINE_STEP = 0.10     # 分数位移粗搜步长(样本)，细搜再除以 10
GROW_MAX = 12.0      # 材料两端按波形向外生长的上限(秒)。实测「播报+材料开头」粘连
                     # 最多 7 秒（D04: 3.10+3.93）
BOUNDARY_WIN = 1.0   # 切点核对窗口(秒)
SILENT_RMS = 0.0178  # = -35dB，与 detect_silences_np 的静音门限一致：低于它的窗口按静音
                     # 处理（两声静音相比只剩底噪，残差随机；提示音另由 marker_set 排除）
PROBE_MIN_RMS = 0.0356  # ≈ -29dB，= 2×SILENT_RMS。两边都低于它时「判不了」而不是
                        # 「对不上」：一边室内底噪、一边纯静音的比值恒为 1.00（见
                        # _probe_resid 的实测），真多带内容时人声 rms≈0.10 抓得住
PROBE_VOICED = 0.5      # 切点核对时，窗口里至少要有这么大比例的 50ms 小格有声音，
                        # 否则判不了（多半是复制过来的静音掺了点串音，见 _voiced_frac）
ANCHOR_SPAN = 2      # 重锚配对边界时，起点/终点附近各搜几块
PAIR_CORR = 0.75     # 配对候选的粗筛门限（单块相关度）
RESCUE_CORR = 0.30   # 兜底通道的门限，比主通道低：有些制品把同一段材料的两遍做过
                     # 变速/剪辑，单块相关度只剩 0.4~0.6，主通道就整段漏掉了
                     # （无关音频的单块相关度 <0.1，0.30 仍有足够余量）
RESCUE_SPAN = 4      # 兜底通道只在开头对上，边界要在附近多搜几块才能定出来
RESCUE_RESID = 1.0   # 兜底通道「退回时长链」时的波形底线：只挡完全不像的两段。
                     # 两道不同题的时长也可能碰巧接近 —— 实测 单元检测卷（5）把
                     # 第 1 题(60.15~73.44)和第 2 题(87.18~99.28)凑成一对，整段残差
                     # 1.30；变速/剪辑过的两遍残差会偏高，所以这里放到 1.0 而不是 0.35
EST_WIN = 3.0        # 短窗 lag 估计的模板长度(秒)，见 estimate_lag 的实测说明
EST_POS = (0.5, 0.3, 0.7)   # 模板在第一遍跨度里的取样位置（比例），取最吻合的那个
PAIR_LAG_TOL = 2.0   # 配对确认容许「两遍的起点错开」的最大秒数（实测见过 1.6 秒）
CHUNK_RESID = 0.60   # 配对确认的单块残差门限：真配对 ≤0.12，无关片段 ≥1.0（实测）


def match_pct(resid):
    """把两遍的「残差」换算成老师看得懂的吻合度百分比。

    残差是两遍波形相减后剩下的杂音比例：两遍一模一样 → 0；两遍毫不相干 → 1.414。
    换算成百分比就是：一模一样 100%，毫不相干 0%。
    """
    if resid is None:
        return None
    return max(0.0, (1.0 - resid / 1.4142) * 100.0)


def frac_shift(v, d):
    """把 v 平移 d 个样本（d 可正可负、可带小数），即带限分数延迟。

    【为什么需要它】流水线把 44.1kHz 的原文件解码成 8kHz，重采样之后两遍的偏移
    不再是整数个 8kHz 样本。只按整数样本对齐会让残差虚高 2~4 倍，而且每段虚高的
    倍数还不一样 —— 实测同一份录音同一段材料：D10 0.344→0.099、D02 0.218→0.087、
    D07 0.323→0.079（换算成「两遍吻合」就是从 76%~85% 变成 93%~94%）。残差既是
    报数也是采纳/拒绝的判据（RESID_OK），虚高会把真材料的开头/结尾误判成「对不上」
    —— 实测真材料开头的扩段只有 0.18~0.20，离阈值 0.35 只剩 0.15 的余量。
    """
    if d == 0.0 or len(v) == 0:
        return v
    n = len(v)
    return np.fft.irfft(np.fft.rfft(v) * np.exp(-2j * np.pi * np.arange(n // 2 + 1) * d / n), n)


def fine_shift(x, s, e, lag):
    """在整数样本的 lag 附近，把分数位移精化到 0.02 个样本（成本约 5ms，18 次评估）。

    返回的是**样本数**（可正可负、绝对值 ≤ FINE_RANGE），调用方要 /SR 化成秒。

    拿 [s, s+FINE_WIN] 与它平移 lag 之后的区间比，找让残差最小的分数位移。
    窗口不取长：实测这个分数位移是全段常量（D01 前后各 12 秒都是 -0.10 样本，
    D10 是 0.48/0.50），几秒足够；窗口太短（<0.5 秒）就放弃精化。
    """
    n = int(min(FINE_WIN, e - s) * SR)
    if n < SR // 2:
        return 0.0
    i0 = int(round(s * SR))
    j0 = i0 + int(round(lag * SR))
    if i0 < 0 or j0 < 0 or j0 + n > len(x):
        return 0.0
    a = x[i0:i0 + n]
    den = float(np.sqrt((a * a).mean()))
    if den < 1e-6:
        return 0.0
    b = x[j0:j0 + n]

    def r(d):
        return float(np.sqrt(((a - frac_shift(b, -d)) ** 2).mean())) / den

    d = min(np.arange(-FINE_RANGE, FINE_RANGE + 1e-9, FINE_STEP), key=r)
    return float(min(np.arange(d - FINE_STEP, d + FINE_STEP + 1e-9, FINE_STEP / 5), key=r))


def slice_resid(x, s, e, lag):
    """比较 [s,e] 与它整体平移 lag 秒后的区间，返回相对残差 rms(diff)/rms(信号)。

    「播两遍」是同一段数字音频复制粘贴两次，所以材料主体内的残差应远小于 1；
    若材料两端混进了只播一遍的东西（题目播报、答题提示），残差会在那里突增。

    lag 允许带小数样本（find_lag 给的就是亚样本级），小数部分用 FFT 相位旋转补上：
    少了这一步，亚样本精化等于白做。
    """
    n = int(round((e - s) * SR))
    if n < SR // 10:
        return None
    i0 = int(round(s * SR))
    j0f = i0 + lag * SR
    j0 = int(np.floor(j0f))
    if i0 < 0 or j0 < 0 or i0 + n > len(x) or j0 + n + 1 > len(x):
        return None
    a = x[i0:i0 + n]
    # frac_shift(v,d) 把 v 延后 d 个样本；这里要把第二遍窗口往前取到 j0f，
    # 所以要传 -d（先前写成 +d，整体错开一个样本，残差直接虚高到 1.3）
    b = frac_shift(x[j0:j0 + n + 1], j0 - j0f)[:n]
    den = float(np.sqrt((a * a).mean()))
    if den < 1e-6:
        return None
    return float(np.sqrt(((a - b) ** 2).mean()) / den)


def align_pair(x, s1, e1, s2, e2, max_shift=ALIGN_SHIFT):
    """求两遍的时间偏移，并据此量化「两遍是否一致」。

    ① 在 p2 附近做归一化 FFT 互相关，求出整数样本精度的 lag
    ② 主体残差 = 第一遍与它平移 lag 后的版本的差异，作为本段的置信度
    ③ 两端残差 / 主体残差 = 局部抬升比。多切进来的内容只在某一遍里出现，
       两端就会明显高于主体，这个比值才是指示边界错误的东西

    【为什么不用残差绝对值判边界】它首先反映的是这份录音的两遍复制质量，
    8 份样本实测从 0.02 到 0.35 差了十倍以上，拿绝对值当阈值会把「录音差点」
    误报成「切错了」。

    【曾经有、后来删掉的东西】这里原本还有一段「沿两端生长」的边界自校正：
    只要相邻区间有声音且两遍一致就把边界往外扩。做消融实验发现它一次都没
    触发过——即使把 extend_pair_ends / merge_pair_fragments 全部关掉，
    开与不开的结果逐文件完全一致。纯投机，已删。

    返回 (resid_body, edge_ratio, lag)，失败返回 None。lag 是亚样本级的偏移，
    调用方拿它去核对切点（boundary_check）时口径才一致。
    """
    n1 = int(round((e1 - s1) * SR))
    if n1 < SR:
        return None
    lag, _raw = find_lag(x, s1, e1, s2, e2, max_shift)
    if lag is None:
        return None
    # 报数用亚样本对齐后的残差：find_lag 的 lag 已经含分数位移，slice_resid 会
    # 把它算进对齐里，于是同一段材料在不同解码/相位下得到同一个数（实测十段
    # 从 75~95% 收敛到 90~95%）。判据用的仍是 find_lag 的整数口径 resid。
    resid = slice_resid(x, s1, e1, lag)
    if resid is None or resid < 1e-6:
        return resid, None, lag
    win = min(EDGE_WIN, (e1 - s1) / 4.0)
    edges = [v for v in (slice_resid(x, s1, s1 + win, lag),
                         slice_resid(x, e1 - win, e1, lag)) if v is not None]
    # 两端残差本身不到 EDGE_MIN 就不算「局部抬升」：比值会骗人（见 EDGE_MIN 的实测）
    top = max(edges) if edges else None
    return resid, (top / resid if top is not None and top >= EDGE_MIN else None), lag


def estimate_lag(x, s1, e1, s2, e2, max_shift=ALIGN_SHIFT, win=EST_WIN):
    """求两遍的时间偏移（亚样本精度），但只用第一遍里的一小段当模板。

    【为什么不再拿整段当模板】旧版把第一遍整段（实测最长 80 秒）当模板做互相关，
    一次要跑 2M 点的 FFT（实测 22ms），而 reanchor 对 25 个边界组合各做一次，
    单份录音光这一步就 8.9 秒（001 全程 10.6 秒）。可两遍的偏移是整段常量 —— 实测
    同一段材料前后各 12 秒搜出来的最优位移完全相同 —— 取中部 3 秒足够定出同一个
    位移，FFT 规模从 2M 点降到 128K 点，单次约 2ms。

    三个取样位置（中部 / 偏前 / 偏后）依次试，第一个够吻合的就返回：材料中部偶尔
    落在换气停顿上，那一段几乎没有内容，峰值不可靠（判据是窗口 rms 高于静音门限）。

    返回 (lag, resid)：lag 是亚样本级的「第二遍起点 − 第一遍起点」（秒），resid 是
    模板窗口上的**整数样本口径**残差，只用来判断这两个位置是不是同一段内容。
    跨度太短、模板没内容、或搜索窗越界返回 (None, None)。
    """
    span = e1 - s1
    if span < 1.0:
        return None, None
    i0_all = int(round(s1 * SR))
    n_tot = int(round(span * SR))
    w = int(min(win, span) * SR)
    if w < SR // 2 or i0_all < 0 or i0_all + n_tot > len(x):
        return None, None
    d0 = int(round((s2 - s1) * SR))          # 名义位移：搜索窗跟着模板走
    for frac in EST_POS:
        off = int((n_tot - w) * frac)
        a = x[i0_all + off:i0_all + off + w]
        sa = float(np.dot(a, a))
        if sa < 1e-12 or float(np.sqrt(sa / w)) < SILENT_RMS:
            continue
        i0 = i0_all + off
        lo = max(0, i0 + d0 - int(max_shift * SR))
        hi = min(len(x), i0 + d0 + int(max_shift * SR) + w)
        b = x[lo:hi]
        if len(b) < w:
            continue
        lags = len(b) - w + 1
        nfft = 1 << (len(b) + w).bit_length()
        corr = np.fft.irfft(np.fft.rfft(b, nfft) * np.conj(np.fft.rfft(a, nfft)),
                            nfft)[:lags]
        cs = np.concatenate([[0.0], np.cumsum(b * b)])
        norm2 = np.maximum(cs[w:] - cs[:lags], 1e-12)
        k0 = int(np.argmax(corr / np.sqrt(norm2)))
        lag_int = (lo - i0 + k0) / SR
        # 分数位移沿用 fine_shift（返回样本数），口径与旧版 find_lag 一致
        d = fine_shift(x, i0 / SR, (i0 + w) / SR, lag_int)
        num = sa + float(norm2[k0]) - 2.0 * float(corr[k0])
        return lag_int + d / SR, float(np.sqrt(max(num, 0.0) / sa))
    return None, None


def span_resid_int(x, s, e, lag):
    """整段残差（**整数样本对齐**，即 RESID_OK / RESCUE_* 标定时用的老口径），不做 FFT。

    直接用展开式 Σ(a−b)² = Σa² + Σb² − 2Σab：三个点积就够，没有互相关、也没有分数
    位移。判据必须留在这个口径上 —— 换成亚样本口径等于换标定，实测会让二十来份
    样本的配集成片漂移（17-标速(美音2) D06 报数 0.040→1.364）。报数另由 slice_resid
    用亚样本口径算，两者取 min。

    第一遍比第二遍剩余空间长时按能比的长度截断 —— 旧版 find_lag 在这里直接放弃，
    恰恰放过「第一遍多带了一截」这种必须测出来的情况；截断后不足原长一半就判不了。
    """
    n = int(round((e - s) * SR))
    if n < SR // 10:
        return None
    i0 = int(round(s * SR))
    j0 = i0 + int(round(lag * SR))
    if i0 < 0 or j0 < 0 or i0 + n > len(x):
        return None
    n2 = min(n, len(x) - j0)
    if n2 < n // 2 or n2 < SR // 10:
        return None
    a = np.asarray(x[i0:i0 + n2], dtype=np.float64)
    b = np.asarray(x[j0:j0 + n2], dtype=np.float64)
    sa = float(np.dot(a, a))
    if sa < 1e-12:
        return None
    return float(np.sqrt(max(sa + float(np.dot(b, b)) - 2.0 * float(np.dot(a, b)), 0.0) / sa))


def find_lag(x, s1, e1, s2, e2, max_shift=ALIGN_SHIFT):
    """求第一遍相对第二遍的时间偏移（亚样本精度），返回 (lag, resid)。

    lag 由 estimate_lag 在短窗上估出来（快），resid 由 span_resid_int 按整段、整数
    样本口径算（判据的标定口径）。两者分开是这一版最大的提速点：旧版拿整段当模板
    做 FFT 互相关，一次 22ms，而 reanchor 要调用 25 次。

    lag = 第二遍起点 - 第一遍起点。resid 见 slice_resid：两遍完全相同 → 0，
    毫不相干 → 1.414。窗口不够长返回 (None, None)。

    lag 定义在**样本域**（i0 是第一遍窗口的样本号、b0 是搜索窗的样本号、k 是峰值）：
    lag = (b0 - i0 + k)/SR。这一点很要紧 —— 如果拿 s2-max_shift 这个没取整的秒数当
    基准，lag 会自带 0.5 个样本以内的参考误差；旧版只按整数样本比残差、取整时正好
    抵消掉了，改用亚样本对齐以后它就会把分数位移顶掉（实测 D02 因此差了 0.6 个样本、
    残差从 0.079 虚高到 0.216）。
    """
    lag, _win = estimate_lag(x, s1, e1, s2, e2, max_shift)
    if lag is None:
        return None, None
    return lag, span_resid_int(x, s1, e1, round(lag * SR) / SR)


def _span_has_marker(chunks, marker_set, t0, t1):
    """[t0,t1) 里有没有提示音块：提示音只播一遍，不该被并进材料"""
    return any(chunks[i][1] > t0 and chunks[i][0] < t1 for i in marker_set)


def grow_bounds(x, chunks, marker_set, s, e, lag, max_grow=GROW_MAX):
    """按波形把第一遍的起止点往外推，返回 (s, e)。

    两遍是同一份录音、偏移恒定，所以第一遍比现在多出来的部分，必须在第二遍对应
    位置也能找到。只播一遍的「播报」「答题提示」一挪进来残差就爆掉：实测采纳的
    扩段 0.05~0.06、拒绝的 0.62~2.55，阈值沿用 RESID_OK。

    【为什么需要它】材料的开头会被两件事吃掉，配对整个都从材料中间开始：
      ① 播报与材料开头之间没有 ≥0.6 秒静音，被静音检测粘成同一块 —— 本样本 D02
         的真起点 139.076 落在块 19（135.580~141.339）内部；
      ② 材料开头那块在第二遍里被静音切成两块，配对用的「块时长之和」对不上 ——
         本样本 D04 第一遍块 35 是 3.934 秒整块，第二遍是 2.334+0.892。
    实测把边界长回去以后：D02 → 139.076（扩段残差 0.195→0.054）、
    D04 → 232.194（0.181→0.061），第一节 5 段时长变成 14.1/14.1/13.9/15.0/17.6 秒。

    候选边界只用真实静音边界：本侧块边界，或对侧块边界 ∓ lag（后者可以落在块中间，
    D02 的 139.076 就是「第二遍块 22 的起点 − lag」）。取通过校验的最长候选。
    向外长的总量受「第二遍起点 − 第一遍终点」限制（两遍区间不能重叠）。

    候选只落在真实静音边界上（试过按 0.25 秒步长连续细扫，材料会顺着被复制的停顿
    一路长出去 —— 实测第一套听力十段每段多带 0.75 秒静音；这也正是上一版「沿两端
    生长」被删掉的原因，所以不再放开这条约束）。

    尾部用同一套判据对称处理。

    注：align_pair 的 docstring 里记着上一版「沿两端生长」的代码因消融实验一次都
    没触发而被删；这一次的触发证据是 D02/D04 两处（扩段残差 0.18~0.20 → 0.05~0.06）
    加上回归里逐条审计过的若干处长回。
    """
    s0 = s
    budget = s + lag - e           # 第一遍终点到第二遍起点之间的余量(秒)
    if budget <= 0:
        return s, e
    # ---- 头部 ----
    # 长完一轮再长一轮：候选窗口是跟着边界挪的（`s - max_grow` 起步），一轮只能看到
    # 12 秒内的候选；实测 临沂 D10 一轮停在 874.737，再跑一轮才够到 872.097。
    for _round in range(8):
        before = s
        cands = [t for t, _ in chunks if s - max_grow <= t < s - 1e-9]
        cands += [t - lag for t, _ in chunks if s + lag - max_grow <= t < s + lag - 1e-9]
        for t in sorted(set(cands)):
            if t < max(0.0, s0 - budget) or _span_has_marker(chunks, marker_set, t, s):
                continue
            r = _probe_resid(x, chunks, marker_set, t, s, lag)
            if r is not None and r <= RESID_OK:
                s = t
                break
        if s >= before - 1e-9:
            break
    # ---- 尾部 ----
    for _round in range(8):
        before = e
        cands = [b for _, b in chunks if e < b <= e + max_grow]
        cands += [b - lag for _, b in chunks if e + lag < b <= e + lag + max_grow]
        for t in sorted(set(cands), reverse=True):
            if t <= e or t > s + lag or _span_has_marker(chunks, marker_set, e, t):
                continue
            r = _probe_resid(x, chunks, marker_set, e, t, lag)
            if r is not None and r <= RESID_OK:
                e = t
                break
        if e <= before + 1e-9:
            break
    return s, e


def trim_bounds(x, chunks, marker_set, s, e, lag, max_trim=GROW_MAX):
    """把两端不属于材料的东西剪掉（与 grow_bounds 对称），返回新的 (s, e)。

    grow_bounds 是「外面那一片吻合就往外长」，这里是「最外面那一片不吻合就往里收」。
    两头都剪 —— 旧版只剪开头，而且是拿「遇到提示音 / 间隔 ≥5 秒」这两个启发式猜：
    实测 001 的 D01 因此把结尾那声「叮」（87.61~88.63，本身就是一个提示音块）留在
    材料里，成品以一声叮结尾、切点核对也必炸；单元检测卷（5）的 D01 又被「≥5 秒
    间隔」一路推到最后一块，只剩 1.223 秒。

    判据是波形：最外侧那一片（必须落在真实静音边界上，绝不放开成固定步长细扫）
    与第二遍对不上就往里收。三种情况不收：
      ① 收不下去了（剩余不足 MIN_PAIR_SPEECH、或不足原跨度一半）；
      ② 最外侧那片里含提示音块 —— 这条不看残差直接收，提示音本来就不该在材料里
         （同一份录音里的提示音天然互相吻合，残差判据对它们无效）；
      ③ 两边都听不出内容（一边是室内底噪、另一边是纯静音，比值恒为 1.00，判不了）。
    """
    if lag is None:
        return s, e
    orig = e - s
    floor = max(MIN_PAIR_SPEECH, orig * 0.5)
    for _round in range(24):
        s0, e0 = s, e
        # ---- 开头：往右收 ----
        cands = [a for a, _ in chunks if s < a <= s + max_trim]
        cands += [a - lag for a, _ in chunks if s < a - lag <= s + max_trim]
        for t in sorted(set(c for c in cands if c > 0)):
            if t <= s + 1e-9 or e - t < floor:
                continue
            r = (_span_has_marker(chunks, marker_set, s, t) or
                 _probe_resid(x, chunks, marker_set, s, t, lag))
            if r is True or (r is not None and r > RESID_OK):
                s = t
                break
            if r is None:
                continue        # 判不了（大多是一片静音），再往里看一片
            break               # 这片是材料，开头到这儿为止
        # ---- 结尾：往左收 ----
        cands = [b for _, b in chunks if e - max_trim <= b < e]
        cands += [b - lag for _, b in chunks if e - max_trim <= b - lag < e]
        for t in sorted(set(c for c in cands if c > 0), reverse=True):
            if t >= e - 1e-9 or t - s < floor:
                continue
            r = (_span_has_marker(chunks, marker_set, t, e) or
                 _probe_resid(x, chunks, marker_set, t, e, lag))
            if r is True or (r is not None and r > RESID_OK):
                e = t
                break
            if r is None:
                continue
            break
        if s <= s0 + 1e-9 and e >= e0 - 1e-9:
            break
    return s, e


def _probe_resid(x, chunks, marker_set, t0, t1, lag, strict_quiet=False):
    """[t0,t1] 与它平移 lag 之后的区间的残差；窗口越界/基本是静音/含提示音返回 None

    静音窗口必须排除：两声静音相比只剩房间底噪，残差是随机的，拿它当「吻合」
    会让材料边界靠静音往外长（这些制品的停顿有时是同一段静音复制出来的）。

    提示音窗口也必须排除：提示音本来就是同一段录音反复播放，两遍的提示音天然
    吻合（实测 镇江/南通 D06 切点外侧 1 秒残差 0.19，那 1 秒正是提示音），拿它
    当「外面还有材料」会误报。

    【strict_quiet】：要不要把「一边是室内底噪、另一边是纯数字静音」也算判不了。
    这两种窗口的比值都恒等于 1.00，光看残差分不出来，但用途不同 ——
      · 修剪/生长（strict_quiet=False，老口径）：只要第一遍这边有声音就照判。
        实测 001 的 D06 开头那 1.98 秒题目播报 rms 只有 0.0328（比人声低 10dB），
        第二遍那边是 0.0170；按老口径算出来残差 0.86，正是要剪掉的东西。
        放宽了反而会被挡住、剪不动。
      · 切点核对（strict_quiet=True）：20 份高二报纸听力的材料尾巴上有 1.5 秒
        室内底噪（rms≈0.026，约 −31.8dB），第二遍剪得干净是纯静音（rms=0.0000），
        于是几乎每段都被打成「切点内侧对不上（残差 1.02~1.40）」。这种只差一点
        底噪的事不值得惊动老师，两边都低于 PROBE_MIN_RMS 就判不了、不打招呼；
        真多带内容时第一遍那边是实打实的人声（rms≈0.10），照样抓得住。
      · 同一道闸还有第二层：只看整窗 rms 挡不住「大半是复制过来的静音、只掺了
        一两百毫秒声音」的窗口（实测 004 D05 切点外侧 1 秒 = 0.8 秒静音 + 0.2 秒
        下一题的播报，整窗 rms 0.0595 已过线，而静音是同一段复制的、残差 0.03，
        于是报「外面还有材料」）。故再要求窗口里至少一半的 50ms 小格有声音，
        否则同样判不了。
    """
    if _span_has_marker(chunks, marker_set, t0, t1):
        return None
    n = int(round((t1 - t0) * SR))
    i0 = int(round(t0 * SR))
    if n < SR // 10 or i0 < 0 or i0 + n > len(x):
        return None
    rms_a = float(np.sqrt((x[i0:i0 + n] ** 2).mean()))
    if rms_a < SILENT_RMS:
        return None
    if strict_quiet and lag is not None:
        j0 = i0 + int(round(lag * SR))
        if 0 <= j0 and j0 + n <= len(x):
            rms_b = float(np.sqrt((x[j0:j0 + n] ** 2).mean()))
            if max(rms_a, rms_b) < PROBE_MIN_RMS:
                return None
            if max(_voiced_frac(x, i0, n), _voiced_frac(x, j0, n)) < PROBE_VOICED:
                return None
    return slice_resid(x, t0, t1, lag)


def _voiced_frac(x, i0, n, frame=SR // 20):
    """窗口里有多大比例的 50ms 小格算「有声音」（rms ≥ SILENT_RMS）。

    整窗 rms 会被「一小段很响 + 一大段静音」拉过线，所以还要看密度：真材料窗
    口几乎每格都有声，而材料外面那种「复制过来的静音 + 一点点串音」只有一两成。
    """
    m = (n // frame) * frame
    if m <= 0:
        return 0.0
    fr = x[i0:i0 + m].reshape(-1, frame)
    return float((np.sqrt((fr * fr).mean(axis=1)) >= SILENT_RMS).mean())


def boundary_check(x, chunks, marker_set, s, e, lag):
    """核对切点，返回 (内头, 外头, 内尾, 外尾) 四个残差（None = 静音/越界/含提示音）。

    这是不看「两遍吻合」百分比的独立检查：切点内侧应是材料（与第二遍吻合），
    外侧不该是材料（与第二遍对不上）。拿它跑旧版输出，D02（起点被切到 142.161）
    会当场露馅：外侧 [141.161,142.161] 与第二遍仍然吻合，说明外面还有属于材料的
    内容 —— 也就是开头被切掉了。所以这个检查对 D02/D04 那类缺陷是有牙齿的。

    这里走 strict_quiet=True：两边都是底噪级（一边室内底噪、一边纯静音）时判不了、
    不打招呼，免得整份卷子每段都被标感叹号（实测 20 份高二报纸听力因此满屏告警）。
    """
    w = BOUNDARY_WIN
    return tuple(_probe_resid(x, chunks, marker_set, t0, t1, lag, strict_quiet=True)
                 for t0, t1 in ((s, s + w), (s - w, s), (e - w, e), (e, e + w)))


def boundary_complaint(check, win=BOUNDARY_WIN):
    """把切点核对的四个残差翻成中文提醒；都正常返回空串。

    内外两侧共用 RESID_OK：内侧 ≤0.35 视为「确实是材料」，外侧 >0.35 视为「不是材料」。
    """
    names = ("切点内侧", "开头外侧", "切点内侧", "结尾外侧")
    for v, nm, inner in zip(check, names, (1, 0, 1, 0)):
        if v is None:
            continue
        if inner and v > RESID_OK:
            return (f"{nm} {win:.0f} 秒与它的第二遍对不上（残差 {v:.2f}），"
                    f"可能没对齐或切进了别的内容，请人工听一下")
        if not inner and v <= RESID_OK:
            return (f"{nm} {win:.0f} 秒仍然与它的第二遍吻合（残差 {v:.2f}），"
                    f"说明还有属于材料的内容被切在外面，请人工听一下")
    return ""


def _snap_chunk(chunks, t, lo):
    """离时刻 t 最近的语音块起点（从第 lo 块往后找），把对齐结果落回块号"""
    return min(range(lo, len(chunks)), key=lambda i: abs(chunks[i][0] - t))


def _snap_chunk_end(chunks, t, lo):
    """离时刻 t 最近的语音块结束时刻，返回其块号 + 1"""
    return min(range(lo, len(chunks)), key=lambda i: abs(chunks[i][1] - t)) + 1


def anchor_pair(x, chunks, i1, j1, i2, j2, marker_set, span=ANCHOR_SPAN):
    """把候选配对的边界重锚到「整段两遍真的吻合」的位置。

    find_repeat_pairs 用「语音块时长链」提出候选，而同一段材料两遍的静音切分
    常常不同，时长链会把提示音和题目指引一起吞进第一遍 —— 于是「单块相关度
    ≈1、整段却对不上」。单块证据能骗过确认，整段残差骗不过（实测假配对的整段
    残差 0.42~1.06，真配对 ≤0.31），所以这里以对齐后的整段残差为准，并在候选
    边界附近各搜几块，取残差最小的那组。

    候选原样就达标时原封不动返回：本来正常的卷子逐段零变化。

    【三项都不达标就返回 None】旧版写成 `reanchor(...) or (i1, j1, i2, j2)`，兜底把
    候选原样放回去，于是「整段怎么对都对不上」的假配对照样过关 —— 单元检测卷（5）
    那对 (9,17,16,22)（第一遍横跨第一节 3 段材料、第二遍横跨 2 段）就是这样活下来的，
    最终被下游剪成 1.223 秒的 D01。旧版 find_lag 在第二遍搜索窗短于第一遍时直接放弃
    （返回 None），连「算一下到底有多不像」的机会都没有，这个兜底等于常开；现在
    span_resid_int 会按能比的长度截断着算，算出来是 0.9 这种数，假配对就拦得住了。
    """
    lag0 = chunks[j1][0] - chunks[i1][0]     # 错位只改变边界，几乎不改变两遍的间隔
    _lag, resid = find_lag(x, chunks[i1][0], chunks[i2 - 1][1],
                           chunks[j1][0], chunks[j2 - 1][1])
    if resid is not None and resid <= RESID_OK:
        return i1, j1, i2, j2
    return reanchor(x, chunks, i1, j1, i2, j2, lag0, marker_set, span)


def reanchor(x, chunks, i1, j1, i2, j2, lag0, marker_set, span):
    """在候选起点/终点附近各搜几块，返回整段最吻合的那组边界；都不达标返回 None。

    只认波形：拿 [s1, e1] 与它平移 lag 之后的区间算整段残差，不看时长链，
    所以两遍被静音切成不同块数（甚至不同块长）也能对齐。

    【lag 只估一次】两遍的偏移是整段常量（见 estimate_lag 的实测），旧版却给 25 个
    边界组合各估一次 —— 既慢（单份 8.9 秒），又让不同组合拿到不同 lag、给「取最长」
    的择优掺进噪声。这里先用 estimate_lag 在候选原跨度上定出 lag，所有组合共用；
    只有候选跨度本身被时长链带偏、中部都不是真材料时，才退回名义位移 lag0 再试一次。
    打分用 span_resid_int（整段、整数样本口径，不含 FFT），与旧版同口径。
    """
    lag, _w = estimate_lag(x, chunks[i1][0], chunks[i2 - 1][1],
                           chunks[j1][0], chunks[j2 - 1][1])
    lags = []
    for lg in (lag, lag0):
        if lg is not None and not any(abs(lg - u) < 0.02 for u in lags):
            lags.append(lg)
    best = None
    for lg in lags:
        for a in range(max(0, i1 - span), min(i1 + span + 1, j1)):
            if a in marker_set:
                continue
            for b in range(max(a + 1, i2 - span), min(len(chunks), i2 + span + 1)):
                if b - 1 in marker_set:
                    continue
                s1, e1 = chunks[a][0], chunks[b - 1][1]
                r = span_resid_int(x, s1, e1, lg)
                # 达标的组合里取最长的那段：宁可从宽（多带的会在下游打 CHECK），
                # 也不要把一截材料拦腰切掉。同样长时才比谁更吻合。
                if r is None or r > RESID_OK:
                    continue
                key = (e1 - s1, -r)
                if best is None or key > best[0]:
                    best = (key, a, b, lg)
    if best is None:
        return None
    _key, a, b, lag = best
    c = _snap_chunk(chunks, chunks[a][0] + lag, b)
    # 重锚后第一遍末尾与第二遍起点之间若还夹着语音块（同一段材料的两截被静音
    # 切开了），按小停顿并进第一遍；否则那半截材料会被切掉
    while (b < c and (b - 1) not in marker_set
           and chunks[b][0] - chunks[b - 1][1] <= 2.0):
        b += 1
    d = max(c + 1, _snap_chunk_end(chunks, chunks[b - 1][1] + lag, c))
    return a, c, b, d


# ---------- assemble materials ----------

def extract_materials(x, chunks, markers, pairs):
    """exam format: section 1 = 5 single-play texts, section 2 = 5 double-played texts"""
    marker_set = set(markers)
    notes = []
    pairs = extend_pair_ends(x, chunks, marker_set, pairs)
    pairs = merge_pair_fragments(chunks, pairs)

    mat_pairs, disc = [], []
    for i1, j1, i2, j2 in pairs:
        gap = chunks[j1][0] - chunks[i2 - 1][1]
        total = chunks[i2 - 1][1] - chunks[i1][0]
        if gap <= MAX_PLAY_GAP and total >= MIN_PAIR_SPEECH:
            mat_pairs.append((i1, j1, i2, j2, gap))
        else:
            disc.append((i1, j1, i2, j2, gap))

    pair_zone = set()
    materials = []
    for i1, j1, i2, j2, gap in mat_pairs:
        for t in range(i1, i2):
            pair_zone.add(t)
        for t in range(j1, j2):
            pair_zone.add(t)
        s = chunks[i1][0]
        e = chunks[i2 - 1][1]
        lag, _ = find_lag(x, s, e, chunks[j1][0], chunks[j2 - 1][1])
        if lag is None:
            al = align_pair(x, s, e, chunks[j1][0], chunks[j2 - 1][1])
        else:
            # 先把不属于材料的两头剪掉（提示音、题目指引、只在一遍里出现的内容），
            # 再按波形往外长 —— 只长不剪的话，时长链吞进来的那两秒会一直留在材料里。
            s, e = trim_bounds(x, chunks, marker_set, s, e, lag)
            # 生长把起止点往外挪了 Δ，第二遍的窗口得跟着挪同样的量：否则 align_pair
            # 在 chunks[j1][0] ±2 秒里搜不到真的第二遍（D04 差了 4.6 秒）
            gs, ge = grow_bounds(x, chunks, marker_set, s, e, lag)
            al_g = align_pair(x, gs, ge, gs + lag, ge + lag) if (gs, ge) != (s, e) else None
            # 长完必须整段仍然对得上，否则只说明「扩进来那一小段碰巧吻合」，一律回退：
            # 实测 17-标速(美音2) D06 尾部误长 9.45 秒，报数从 0.040 崩到 1.364；
            # 另有 3 份长到文件尾，第二遍窗口不够长，直接报「找不到第二遍」。
            if al_g is not None and al_g[0] is not None and al_g[0] <= RESID_OK:
                s, e, al = gs, ge, al_g
            else:
                al = align_pair(x, s, e, s + lag, e + lag)
        if al is None:
            notes.append((chunks[i1][0],
                          f"跳过 {chunks[i1][0]:.3f} 秒处这一段："
                          f"找不到它的第二遍，无法自动校验"))
            resid, ratio, lag2 = None, None, None
        else:
            resid, ratio, lag2 = al
        # resid_raw = 只按整数样本对齐的残差，留着对照：它比 resid 虚高 2~4 倍，
        # 正好说明亚样本精化在干什么（也写进 timeline.json）
        raw = None if lag2 is None else slice_resid(x, s, e, round(lag2 * SR) / SR)
        # 报数取「亚样本对齐」与「整数对齐」里更好的那一个：分数位移是在短窗上估的，
        # 对整段未必最优（实测 20 段反而略差，且都在 40% 以下本来就可疑的段上）。
        # 取 min 让「换口径以后报数只会变好」这条性质严格成立。
        if raw is not None and resid is not None:
            resid = min(resid, raw)
        check = (boundary_check(x, chunks, marker_set, s, e, lag2)
                 if lag2 is not None else (None,) * 4)
        flag = ""
        if resid is None:
            flag = "找不到这一段的第二遍，无法自动校验，请人工听一下"
        elif resid > RESID_OK:
            # 两遍对不上时 lag 本身就不可靠，切点核对的那些数字没有意义，只报这一条
            flag = (f"这两遍的波形对不上（吻合度 {match_pct(resid):.0f}%）：多半是"
                    f"录音方把这段重新读了一遍，而不是复制同一段录音，请人工听一下")
        elif ratio is not None and ratio > EDGE_CHECK:
            flag = ("这一段的开头或结尾可能多带了不该有的内容"
                    "（两端和中间不像同一段），请人工听一下")
        # 两遍本身对不上时 lag 就是错的，切点核对的数字只是噪声，别再叠一条
        if resid is None or resid <= RESID_OK:
            bad = boundary_complaint(check)
            if bad:
                flag = f"{flag}；{bad}" if flag else bad
        materials.append({"start": s, "end": e, "plays": 2, "flag": flag,
                          "resid": resid, "resid_raw": raw, "check": check,
                          "lag": lag if lag is not None else None})

    # 只有「两遍真的对得上」的材料才当第一节的终点：一段没通过校验的材料（比如被
    # 时长链凑出来的假材料）会把这个位置一把拽到很前面，于是「答题静音」兜底只看
    # 到那儿之前，第一节剩下的几段全找不到（实测 单元检测卷（5）的假材料落在 109 秒，
    # 把兜底窗口压到 109 秒之前，5 段只认出 2 段）。
    verified = [m["start"] for m in materials
                if m["resid"] is not None and m["resid"] <= RESID_OK]
    first_pair = (min(verified) if verified
                  else min((m["start"] for m in materials), default=1e9))

    labels = [m for m in markers if m not in pair_zone and gap_before(chunks, m) >= 4.0]

    candidates = []
    for li, m in enumerate(labels):
        block_end = chunks[labels[li + 1]][0] if li + 1 < len(labels) else 1e9
        s, e, c, glued = chunks[m][1], chunks[m][1], m + 1, False
        while c < len(chunks) and chunks[c][0] < block_end:
            if c in marker_set:
                break
            if c in pair_zone:
                # 紧挨着一段双遍材料：这块多半是那段材料的「题目指引」，不是单遍材料
                glued = chunks[c][0] - e < 5.0
                break
            e = chunks[c][1]
            if gap_after(chunks, c) >= 5.0:
                break
            c += 1
        if e - s >= 2.0 and not glued:
            candidates.append({"start": s, "end": e, "plays": 1, "flag": ""})

    kept, after = [], []
    for cd in candidates:
        dur_cd = cd["end"] - cd["start"]
        # 「第二节说明 + Text 6 播报」的合并块与真材料长得一模一样，靠数量上限只是
        # 碰巧把它挤掉；一旦某卷少找到一段真材料，它就会顶上来充当 D05。按时长剔除，
        # 实测这类块是 30~39 秒，而 8 份样本里最长的真材料是 24.3 秒。
        if dur_cd > MAX_SINGLE:
            notes.append((cd["start"],
                          f"跳过 {cd['start']:.3f}~{cd['end']:.3f} 秒（{dur_cd:.3f} 秒）："
                          f"比一段材料长得多（材料最长 {MAX_SINGLE:.0f} 秒），"
                          f"通常是「第二节说明」这类播报，不是材料"))
        elif cd["end"] <= first_pair:
            kept.append(cd)
        else:
            # 已经过了第一段双遍材料，这里剩下的都是答题提示或下一题的播报
            after.append(cd)
    if after:
        d0 = after[0]
        notes.append((0.0, f"另有 {len(after)} 个候选块（如 {d0['start']:.3f}~"
                           f"{d0['end']:.3f} 秒）夹在材料之间，是答题提示或下一题的"
                           f"播报，没有当作材料"))
    kept.sort(key=lambda t: t["start"])
    if len(kept) > N_SINGLE:
        for cd in kept[N_SINGLE:]:
            notes.append((cd["start"],
                          f"跳过 {cd['start']:.3f}~{cd['end']:.3f} 秒"
                          f"（{cd['end'] - cd['start']:.3f} 秒）："
                          f"第一节只要 {N_SINGLE} 段，这一条是多余的"))
        kept = kept[:N_SINGLE]
    # ---- 无提示音录音的兜底 ----
    # 有些制品（实测 2024 届河南郑州三模、河南 4 月模拟联考）每段前面不放提示音，
    # 上面靠提示音定位第一节的路子就整个失效（labels=0，第一节一段都找不到）。
    # 改用「答题静音」定位：材料之间隔着约 10 秒的答题静音，而材料内部只有 1 秒
    # 左右的小停顿，按 gap>=SPLIT_GAP 切开后——
    #   郑州:   [开场说明 65s] 16 20 25 29 24 [第二节说明 42s]  → 掐掉两头正好 5 段
    #   4月联考:[开场说明 42s] 36 35 31 10 9                     → 掐掉开头正好 5 段
    # 说明块永远在两头、材料在中间，这是高考录音的固定结构，不是巧合。
    if len(kept) < N_SINGLE and first_pair < 1e8:
        blocks, cur = [], []
        for i, (a, _b) in enumerate(chunks):
            if a >= first_pair:
                break
            if cur and a - chunks[cur[-1]][1] >= SPLIT_GAP:
                blocks.append(cur)
                cur = []
            cur.append(i)
        if cur:
            blocks.append(cur)
        # 去掉过短的零头（提示音残余、换气声）
        blocks = [b for b in blocks
                  if chunks[b[-1]][1] - chunks[b[0]][0] >= MIN_SINGLE]

        def _span(bt):
            return chunks[bt[-1]][1] - chunks[bt[0]][0]

        # 判据：第一节的 N_SINGLE 段材料长度彼此接近，而开场说明、第二节说明
        # 都比它们长得多。取「长度差异最小的连续 N_SINGLE 块」——实测：
        #   郑州:   [65开场 16 20 25 29 24 42说明]  → 选中 16~24 那 5 块 ✓
        #   4月联考:[42 36 35 31 10 9 10 11 11]    → 选中 9~11 那 5 块 ✓
        if len(blocks) >= N_SINGLE:
            best = min(range(len(blocks) - N_SINGLE + 1),
                       key=lambda t: (max(_span(blocks[t + u]) for u in range(N_SINGLE)) /
                                      min(_span(blocks[t + u]) for u in range(N_SINGLE))))
            blocks = blocks[best:best + N_SINGLE]

        if len(blocks) == N_SINGLE:
            kept = [{"start": chunks[b[0]][0], "end": chunks[b[-1]][1],
                     "plays": 1, "flag": "", "fallback": True} for b in blocks]
            notes.append((0.0, f"提示音没能定位第一节，改用「答题静音」兜底："
                               f"第一个双遍材料之前正好切出 {N_SINGLE} 块"))
        # 切不出 N_SINGLE 块不是异常：第一节也播两遍的录音本来就没有单遍材料可找。
        # 真正的结构不符由下面的总段数汇报负责，这里不再另发告警。

    for cd in kept:
        if cd.get("fallback") and first_pair - cd["end"] < 30.0:
            pass                                # 兜底时最后一段本来就紧邻第二节，不算异常
        elif cd["end"] - cd["start"] < MIN_SINGLE:
            cd["flag"] = (f"这一段只有 {cd['end'] - cd['start']:.3f} 秒，"
                          f"比正常的一段材料短得多，可能被切碎了，请人工听一下")
        elif first_pair - cd["end"] < 30.0:
            cd["flag"] = ("这一段紧挨着第二节的第一段，"
                          "开头可能带了第二节的说明，请人工听一下")
    materials.extend(kept)

    # 残差绝对值本身不代表切错了（它首先反映这份录音的两遍复制质量，各卷差 10 倍以上），
    # 但【同一份卷子里明显高于其它段】是个值得人听一下的信号，故按卷内中位数做离群判定。
    rs = sorted(m["resid"] for m in materials if m.get("resid") is not None)
    if len(rs) >= 3:
        med = rs[len(rs) // 2]
        if med > 1e-6:
            for mt in materials:
                r = mt.get("resid")
                if (r is not None and r > OUTLIER_MIN
                        and r > OUTLIER_RESID * med and not mt["flag"]):
                    mt["flag"] = (f"这一段的吻合度（{match_pct(r):.0f}%）明显低于"
                                  f"本卷其它段落（其它段普遍 {match_pct(med):.0f}%）。"
                                  f"可能是这两遍并不是同一份复制（录音方重新录过），"
                                  f"也可能边界不准，请人工听一下")

    materials.sort(key=lambda t: t["start"])
    for i, mt in enumerate(materials, 1):
        mt["n"] = i
    # 提示按时间先后排序输出，方便顺着音频往下看
    return materials, mat_pairs, disc, labels, [t for _, t in sorted(notes)]

# ---------- cut ----------

def cut(src, start, end, out, pad_in=0.10, pad_out=0.35):
    s = max(0.0, start - pad_in)
    dur = end + pad_out - s
    subprocess.run([FF, "-y", "-hide_banner", "-loglevel", "error", "-i", src,
                    "-ss", f"{s:.3f}", "-t", f"{dur:.3f}",
                    "-af", "afade=t=in:st=0:d=0.10", "-c:a", "libmp3lame", "-b:a", "128k", out],
                   check=True)

def max_volume(path, retries=3):
    """max volume in dB, or None if measurement failed (e.g. AV briefly locks
    the just-written file on Windows)"""
    for _ in range(retries):
        r = subprocess.run([FF, "-hide_banner", "-i", path, "-af", "volumedetect", "-f", "null", "-"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        m = re.search(r"max_volume: ([-\d.]+) dB", r.stderr)
        if m:
            return float(m.group(1))
        time.sleep(0.4)
    return None

# ---------- per-file processing ----------

def describe_structure(n_single, n_double):
    """用中文说明这份录音是什么结构。

    旧规则是第一节 5 段播一遍 + 第二节 5 段播两遍；新版（2026 起部分卷）第一节
    也播两遍，于是 10 段全落在「播两遍」这一类里 —— 两者都算正常，不必报警。
    """
    if n_single == N_SINGLE and n_double == N_DOUBLE:
        return f"第一节 {n_single} 段播一遍 + 第二节 {n_double} 段播两遍（标准结构）"
    if not n_single:
        return f"{n_double} 段，每段都播两遍"
    if not n_double:
        return f"{n_single} 段，每段只播一遍"
    return f"{n_single} 段播一遍 + {n_double} 段播两遍"


def process(path, outdir):
    x = decode_all(path)
    dur = len(x) / SR
    chunks = build_chunks(detect_silences(x), dur)
    markers = find_markers(x, chunks)
    pairs = find_repeat_pairs(x, chunks, marker_set=set(markers))
    materials, mat_pairs, disc, labels, notes = extract_materials(x, chunks, markers, pairs)

    n_single = sum(1 for m in materials if m["plays"] == 1)
    n_double = sum(1 for m in materials if m["plays"] == 2)
    print(f"  找到 {len(materials)} 段材料：{describe_structure(n_single, n_double)}")
    for n in notes:
        print(f"    · {n}")
    for i1, j1, i2, j2, gap in disc:
        total = chunks[i2 - 1][1] - chunks[i1][0]
        if gap > MAX_PLAY_GAP:
            print(f"    · 跳过一对疑似重复的片段：{chunks[i1][0]:.3f} 秒 与 "
                  f"{chunks[j1][0]:.3f} 秒相隔 {gap:.3f} 秒，离得太远，"
                  f"不是同一段材料的两遍")
        else:
            print(f"    · 跳过一对疑似重复的片段：{chunks[i1][0]:.3f} 秒处那一段只有 "
                  f"{total:.3f} 秒，太短，不像一段材料")
    if not labels:
        print("    · 注意：没找到「提示音」，第一节可能一段都没定位到，请人工核对")
    for mt in materials:
        extra = ""
        if mt["plays"] == 2:
            p = match_pct(mt.get("resid"))
            extra = f"  两遍吻合 {p:.0f}%" if p is not None else "  两遍没对上"
        print(f"    D{mt['n']:02d}  {mt['start']:7.3f} 秒 ~ {mt['end']:7.3f} 秒"
              f"（共 {mt['end'] - mt['start']:.3f} 秒，"
              f"播{'两' if mt['plays'] == 2 else '一'}遍）{extra}")
    # 切点核对汇总：这是不看「两遍吻合」百分比的独立检查 —— 内侧应吻合（确实是材料）、
    # 外侧应不吻合（外面不是材料）。旧版 D02/D04 那种「开头被切掉」在这里会露馅。
    chk = [mt for mt in materials if mt.get("check")]
    ins = [v for mt in chk for v in (mt["check"][0], mt["check"][2]) if v is not None]
    outs = [v for mt in chk for v in (mt["check"][1], mt["check"][3]) if v is not None]
    parts = []
    if ins:
        parts.append(f"内侧最差 {max(ins):.2f}（应 < {RESID_OK:.2f}）")
    if outs:
        parts.append(f"外侧最好 {min(outs):.2f}（应 > {RESID_OK:.2f}）")
    elif ins:
        parts.append("外侧都是静音或提示音（没有内容漏在外面）")
    if parts:
        print(f"    切点核对（独立于百分比）：{len(chk)} 段，" + "、".join(parts))
    for mt in materials:
        if mt["flag"]:
            print(f"    !! 请留意 D{mt['n']:02d}：{mt['flag']}")
    if len(materials) < N_SINGLE + N_DOUBLE:
        print(f"    !! 注意：这份录音只找到 {len(materials)} 段，"
              f"高考听力标准是 {N_SINGLE + N_DOUBLE} 段，可能有材料没找到，请人工核对")
    elif len(materials) > N_SINGLE + N_DOUBLE:
        print(f"    · 这份录音比标准高考（{N_SINGLE + N_DOUBLE} 段）多，"
              f"按实际的 {len(materials)} 段使用")

    os.makedirs(outdir, exist_ok=True)
    timeline = {
        "materials": {str(i): {"start": round(mt["start"], 3), "end": round(mt["end"], 3),
                               "flag": mt["flag"],
                               "resid": None if mt.get("resid") is None
                               else round(mt["resid"], 3),
                               # resid 是亚样本对齐后的残差；raw 是只按整数样本对齐的
                               # 残差（旧口径），留着对照用：它虚高 2~4 倍
                               "resid_raw": None if mt.get("resid_raw") is None
                               else round(mt["resid_raw"], 3),
                               "match_pct": None if match_pct(mt.get("resid")) is None
                               else round(match_pct(mt.get("resid")), 1),
                               # 切点核对：内侧(head_in/tail_in)应 <RESID_OK（确实是材料），
                               # 外侧(head_out/tail_out)应 >RESID_OK（外面不是材料）
                               "boundary_check": None if not mt.get("check") else
                               dict(zip(("head_in", "head_out", "tail_in", "tail_out"),
                                        (None if v is None else round(v, 3)
                                         for v in mt["check"])))}
                      for i, mt in enumerate(materials, 1)},
        "questions": {},
        "_meta": {"source": os.path.basename(path), "plays": {
                      str(i): mt["plays"] for i, mt in enumerate(materials, 1)},
                  "diag": {"duration": round(dur, 3), "chunks": len(chunks),
                           "markers": len(markers), "pairs_raw": len(pairs),
                           "pairs_kept": len(mat_pairs), "pairs_dropped": len(disc),
                           "labels": len(labels), "notes": notes}},
    }
    with open(os.path.join(outdir, "timeline.json"), "w", encoding="utf-8") as f:
        json.dump(timeline, f, ensure_ascii=False, indent=1)
    vols = []
    for mt in materials:
        out = os.path.join(outdir, f"D{mt['n']:02d}.mp3")
        cut(path, mt["start"], mt["end"], out)
        vols.append(max_volume(out))
    bad = [i + 1 for i, v in enumerate(vols) if v is not None and v < -60]
    unverified = [i + 1 for i, v in enumerate(vols) if v is None]
    if bad:
        print(f"    !! 第 {bad} 段切出来是静音，请人工核对")
    if unverified:
        print(f"    ? 第 {unverified} 段的音量没能校验（文件刚写完被占用），可自行抽查")
    if not vols:
        print("    !! 一段都没切出来，请确认这个 MP3 是不是标准高考听力")
    elif not bad and not unverified:
        print(f"    全部切好，{len(vols)} 段的音量都正常"
              f"（声音最轻的一段是 {min(vols):.1f}dB，不是静音）")
    print(f"    结果已保存到: {outdir}")
    return materials

# ---------- main ----------

def pause():
    try:
        input("按回车键退出...")
    except EOFError:
        pass


def main():
    global FF, N_SINGLE, N_DOUBLE
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    no_pause = "--no-pause" in sys.argv
    fmt = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--format=")), None)
    if fmt:
        try:
            a, b = fmt.split("+")
            N_SINGLE, N_DOUBLE = int(a), int(b)
        except ValueError:
            print(f"[错误] --format 应写成「单遍数+双遍数」, 例如 5+5, 收到的是: {fmt}")
            if not no_pause:
                pause()
            sys.exit(1)

    print("=" * 52)
    print("  高考英语听力真题 · 自动切分工具")
    print("=" * 52)
    print("  说明：自动适配「第一节播一遍」和「第一节也播两遍」两种录音，"
          "每段材料只切第一遍。")
    print("  「两遍吻合」= 同一段材料两次播放的波形有多像，")
    print("  越接近 100% 说明这段切得越干净；明显低于本卷其它段落时会提醒你核对。")

    FF = resolve_ffmpeg()
    if not FF:
        print("[错误] 没有找到 ffmpeg。")
        print("  解决办法: 在命令行执行  pip install imageio-ffmpeg  后重试")
        if not no_pause:
            pause()
        sys.exit(1)

    # collect input files: dropped files / given paths, else all MP3s beside this program
    files = []
    for a in args:
        if os.path.isdir(a):
            files += sorted(glob.glob(os.path.join(a, "*.mp3")))
        elif a.lower().endswith(".mp3"):
            files.append(a)
        else:
            print(f"[跳过] 不支持的文件: {a}")
    if not files:
        here = os.path.dirname(os.path.abspath(sys.argv[0]))
        files = sorted(glob.glob(os.path.join(here, "*.mp3")))
    if not files:
        print("没有找到 MP3 文件。")
        print("请把听力 MP3 放到本程序所在的文件夹里(或直接拖到本程序图标上)再运行。")
        if not no_pause:
            pause()
        sys.exit(0)

    ok, failed = 0, 0
    for i, f in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] {os.path.basename(f)}")
        try:
            outdir = os.path.join(os.path.dirname(os.path.abspath(f)),
                                  os.path.splitext(os.path.basename(f))[0] + "-切分结果")
            process(f, outdir)
            ok += 1
        except Exception as e:
            print(f"  [失败] {e}")
            failed += 1

    print("\n" + "=" * 52)
    print(f"全部完成: 成功 {ok} 个" + (f", 失败 {failed} 个" if failed else ""))
    print("每个 MP3 旁边都有一个「原名-切分结果」文件夹, 里面是 D01-D10。")
    if not no_pause:
        pause()

if __name__ == "__main__":
    main()

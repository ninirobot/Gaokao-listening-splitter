"""高考英语听力真题自动切分工具

用法（三选一）:
  1. 直接双击运行:  切分本文件(或exe)所在文件夹里的全部 MP3
  2. 把 MP3 拖到本程序图标上: 只切分拖入的文件
  3. 命令行: python split-listening-v3.py <文件或文件夹> [--no-pause]

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


def find_markers(x, chunks, min_cluster=6):
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
    best = max(groups.values(), key=len)
    return sorted(best) if len(best) >= min_cluster else []

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
    confirmed = []
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
        ok = xcorr_best(seg(x, *chunks[ev[0][1]]), seg(x, *chunks[ev[0][2]])) > PAIR_CORR
        if not ok:
            ok = sum(1 for _, ci, cj in ev[:5]
                     if xcorr_best(seg(x, *chunks[ci]), seg(x, *chunks[cj])) > PAIR_CORR) >= 2
        if not ok:
            continue
        # 上面只是单块粗筛（便宜）。时长链提出的边界常常吞进了提示音/题目指引，
        # 单块相关度照样很高，必须再用整段对齐精筛一次，否则会切出一段
        # 「含提示音、少了半段材料」的垃圾。
        anc = anchor_pair(x, chunks, i, j, pr[-1][0] + 1, pr[-1][1] + 1, marker_set)
        if anc is None:
            continue
        a, c, b, d = anc
        used.update(range(a, b))
        used.update(range(c, d))
        confirmed.append((a, c, b, d))
    return sorted(confirmed + rescue_pairs(x, chunks, marker_set, confirmed))


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
OUTLIER_RESID = 2.5  # 段残差 / 本卷残差中位数 超过此值 → 建议抽听
MAX_SINGLE = 26.0    # 单遍材料的最长时长(秒)，超过者为「第二节说明」之类的块
MIN_SINGLE = 8.0     # 单遍材料的最短时长(秒)，短于此多半是切碎了（实测真材料最短 9.8 秒）
SPLIT_GAP = 3.0      # 无提示音兜底时，把「答题静音」认作分界的间隔(秒)
RESID_OK = 0.35      # 「这两遍确实是同一段」的整段残差上限（实测真配对 ≤0.31）
ANCHOR_SPAN = 2      # 重锚配对边界时，起点/终点附近各搜几块
PAIR_CORR = 0.75     # 配对候选的粗筛门限（单块相关度）
RESCUE_CORR = 0.30   # 兜底通道的门限，比主通道低：有些制品把同一段材料的两遍做过
                     # 变速/剪辑，单块相关度只剩 0.4~0.6，主通道就整段漏掉了
                     # （无关音频的单块相关度 <0.1，0.30 仍有足够余量）
RESCUE_SPAN = 4      # 兜底通道只在开头对上，边界要在附近多搜几块才能定出来


def match_pct(resid):
    """把两遍的「残差」换算成老师看得懂的吻合度百分比。

    残差是两遍波形相减后剩下的杂音比例：两遍一模一样 → 0；两遍毫不相干 → 1.414。
    换算成百分比就是：一模一样 100%，毫不相干 0%。
    """
    if resid is None:
        return None
    return max(0.0, (1.0 - resid / 1.4142) * 100.0)


def slice_resid(x, s, e, lag):
    """比较 [s,e] 与它整体平移 lag 秒后的区间，返回相对残差 rms(diff)/rms(信号)。

    「播两遍」是同一段数字音频复制粘贴两次，所以材料主体内的残差应远小于 1；
    若材料两端混进了只播一遍的东西（题目播报、答题提示），残差会在那里突增。
    """
    n = int(round((e - s) * SR))
    if n < SR // 10:
        return None
    i0 = int(round(s * SR))
    j0 = int(round((s + lag) * SR))
    if i0 < 0 or j0 < 0 or i0 + n > len(x) or j0 + n > len(x):
        return None
    a = x[i0:i0 + n]
    b = x[j0:j0 + n]
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

    返回 (resid_body, edge_ratio)，失败返回 None。
    """
    n1 = int(round((e1 - s1) * SR))
    if n1 < SR:
        return None
    lag, resid = find_lag(x, s1, e1, s2, e2, max_shift)
    if lag is None:
        return None
    if resid is None or resid < 1e-6:
        return resid, None
    win = min(EDGE_WIN, (e1 - s1) / 4.0)
    edges = [v for v in (slice_resid(x, s1, s1 + win, lag),
                         slice_resid(x, e1 - win, e1, lag)) if v is not None]
    return resid, (max(edges) / resid if edges else None)


def find_lag(x, s1, e1, s2, e2, max_shift=ALIGN_SHIFT):
    """求第一遍相对第二遍的时间偏移（样本精度），返回 (lag, resid)。

    lag = 第二遍起点 - 第一遍起点。resid 见 slice_resid：两遍完全相同 → 0，
    毫不相干 → 1.414。窗口不够长返回 (None, None)。
    """
    n1 = int(round((e1 - s1) * SR))
    if n1 < SR:
        return None, None
    lo = max(0.0, s2 - max_shift)
    hi = min(len(x) / SR, e2 + max_shift)
    a = x[int(round(s1 * SR)):int(round(s1 * SR)) + n1]
    b = x[int(round(lo * SR)):int(round(hi * SR))]
    if len(b) <= n1:
        return None, None

    lags = len(b) - n1 + 1
    nfft = 1 << (len(b) + n1).bit_length()
    corr = np.fft.irfft(np.fft.rfft(b, nfft) * np.conj(np.fft.rfft(a, nfft)), nfft)[:lags]
    cs = np.concatenate([[0.0], np.cumsum(b * b)])
    norm = np.sqrt(np.maximum(cs[n1:] - cs[:lags], 1e-12))
    lag = (lo - s1) + int(np.argmax(corr / norm)) / SR
    return lag, slice_resid(x, s1, e1, lag)


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
    """
    lag0 = chunks[j1][0] - chunks[i1][0]     # 错位只改变边界，几乎不改变两遍的间隔
    _lag, resid = find_lag(x, chunks[i1][0], chunks[i2 - 1][1],
                           chunks[j1][0], chunks[j2 - 1][1])
    if resid is not None and resid <= RESID_OK:
        return i1, j1, i2, j2
    return reanchor(x, chunks, i1, j1, i2, j2, lag0, marker_set, span) or (i1, j1, i2, j2)


def reanchor(x, chunks, i1, j1, i2, j2, lag0, marker_set, span):
    """在候选起点/终点附近各搜几块，返回整段最吻合的那组边界；都不达标返回 None。

    只认波形：拿 [s1, e1] 与它平移 lag0 之后的区间算整段残差，不看时长链，
    所以两遍被静音切成不同块数（甚至不同块长）也能对齐。
    """
    best = None
    for a in range(max(0, i1 - span), min(i1 + span + 1, j1)):
        if a in marker_set:
            continue
        for b in range(max(a + 1, i2 - span), min(len(chunks), i2 + span + 1)):
            if b - 1 in marker_set:
                continue
            s1, e1 = chunks[a][0], chunks[b - 1][1]
            lag, r = find_lag(x, s1, e1, s1 + lag0, e1 + lag0, max_shift=2.0)
            # 达标的组合里取最长的那段：宁可从宽（多带的会在下游打 CHECK），
            # 也不要把一截材料拦腰切掉。同样长时才比谁更吻合。
            if r is None or r > RESID_OK:
                continue
            key = (e1 - s1, -r)
            if best is None or key > best[0]:
                best = (key, a, b, lag)
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
        # 材料起点 = p1 内最后一个提示音 / 最后一个 ≥5s 间隙之后。
        # 必须 clamp 在 i2-1 以内：否则当最后一块本身就是提示音时，start_c 会被推到
        # i2，切出「结束早于开始」的倒挂区间（潍坊卷挽救回来的那段就是这样崩掉的）。
        start_c = i1
        for t in range(i1, i2):
            if t in marker_set and t + 1 < i2:
                start_c = t + 1
            if t + 1 < i2 and gap_after(chunks, t) >= 5.0 and t + 1 > start_c:
                start_c = t + 1
        s = chunks[start_c][0]
        e = chunks[i2 - 1][1]
        if e - s < 1.0:
            notes.append(f"pair @{chunks[i1][0]:.1f}s 头部修剪过度, 回退到未修剪起点")
            s = chunks[i1][0]
        al = align_pair(x, s, e, chunks[j1][0], chunks[j2 - 1][1])
        if al is None:
            notes.append((chunks[i1][0],
                          f"跳过 {chunks[i1][0]:.0f} 秒处这一段："
                          f"找不到它的第二遍，无法自动校验"))
            resid, ratio = None, None
        else:
            resid, ratio = al
        flag = ""
        if resid is None:
            flag = "找不到这一段的第二遍，无法自动校验，请人工听一下"
        elif ratio is not None and ratio > EDGE_CHECK:
            flag = ("这一段的开头或结尾可能多带了不该有的内容"
                    "（两端和中间不像同一段），请人工听一下")
        materials.append({"start": s, "end": e, "plays": 2, "flag": flag,
                          "resid": resid})

    first_pair = min((m["start"] for m in materials), default=1e9)

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
                          f"跳过 {cd['start']:.0f}~{cd['end']:.0f} 秒（{dur_cd:.0f} 秒）："
                          f"比一段材料长得多（材料最长 {MAX_SINGLE:.0f} 秒），"
                          f"通常是「第二节说明」这类播报，不是材料"))
        elif cd["end"] <= first_pair:
            kept.append(cd)
        else:
            # 已经过了第一段双遍材料，这里剩下的都是答题提示或下一题的播报
            after.append(cd)
    if after:
        d0 = after[0]
        notes.append((0.0, f"另有 {len(after)} 个候选块（如 {d0['start']:.0f}~"
                           f"{d0['end']:.0f} 秒）夹在材料之间，是答题提示或下一题的"
                           f"播报，没有当作材料"))
    kept.sort(key=lambda t: t["start"])
    if len(kept) > N_SINGLE:
        for cd in kept[N_SINGLE:]:
            notes.append((cd["start"],
                          f"跳过 {cd['start']:.0f}~{cd['end']:.0f} 秒"
                          f"（{cd['end'] - cd['start']:.0f} 秒）："
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
            cd["flag"] = (f"这一段只有 {cd['end'] - cd['start']:.0f} 秒，"
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
                if r is not None and r > OUTLIER_RESID * med and not mt["flag"]:
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
    chunks = build_chunks(detect_silences_np(x), dur)
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
            print(f"    · 跳过一对疑似重复的片段：{chunks[i1][0]:.0f} 秒 与 "
                  f"{chunks[j1][0]:.0f} 秒相隔 {gap:.0f} 秒，离得太远，"
                  f"不是同一段材料的两遍")
        else:
            print(f"    · 跳过一对疑似重复的片段：{chunks[i1][0]:.0f} 秒处那一段只有 "
                  f"{total:.0f} 秒，太短，不像一段材料")
    if not labels:
        print("    · 注意：没找到「提示音」，第一节可能一段都没定位到，请人工核对")
    for mt in materials:
        extra = ""
        if mt["plays"] == 2:
            p = match_pct(mt.get("resid"))
            extra = f"  两遍吻合 {p:.0f}%" if p is not None else "  两遍没对上"
        print(f"    D{mt['n']:02d}  {mt['start']:6.1f} 秒 ~ {mt['end']:6.1f} 秒"
              f"（共 {mt['end'] - mt['start']:.0f} 秒，"
              f"播{'两' if mt['plays'] == 2 else '一'}遍）{extra}")
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
        "materials": {str(i): {"start": round(mt["start"], 2), "end": round(mt["end"], 2),
                               "flag": mt["flag"],
                               "resid": None if mt.get("resid") is None
                               else round(mt["resid"], 3),
                               "match_pct": None if match_pct(mt.get("resid")) is None
                               else round(match_pct(mt.get("resid")), 1)}
                      for i, mt in enumerate(materials, 1)},
        "questions": {},
        "_meta": {"source": os.path.basename(path), "plays": {
                      str(i): mt["plays"] for i, mt in enumerate(materials, 1)},
                  "diag": {"duration": round(dur, 2), "chunks": len(chunks),
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

"""高考英语听力真题自动切分工具

用法（三选一）:
  1. 直接双击运行:  切分本文件(或exe)所在文件夹里的全部 MP3
  2. 把 MP3 拖到本程序图标上: 只切分拖入的文件
  3. 命令行: python split-listening.py <文件或文件夹> [--no-pause]

输出: 每个 MP3 旁边生成 <原名>-切分结果/ 文件夹 (D01-D10.mp3 + timeline.json)
"""
import re, subprocess, sys, json, os, glob, shutil, time
import numpy as np

SR = 8000
MAX_PLAY_GAP = 30.0
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

def find_markers(x, chunks, min_cluster=6):
    cand = [i for i, (a, b) in enumerate(chunks) if 0.6 <= b - a <= 2.5]
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

def find_repeat_pairs(x, chunks, tol_abs=0.10, tol_rel=0.04, min_chunks=2, min_total=3.0):
    n = len(chunks)
    durs = [b - a for a, b in chunks]

    def close(a, b):
        return abs(a - b) <= max(tol_abs, tol_rel * max(a, b))

    def match_len(i, j):
        k = 0
        while i + k < j and j + k < n and close(durs[i + k], durs[j + k]):
            k += 1
        return k

    cands = []
    for i in range(n):
        for j in range(i + 1, n):
            if i > 0 and j > 0 and close(durs[i - 1], durs[j - 1]):
                continue
            k = match_len(i, j)
            tot = sum(durs[i:i + k])
            # single-chunk matches allowed only for long chunks (material with no
            # internal silence is one chunk per play)
            if tot >= min_total and (k >= min_chunks or (k == 1 and tot >= 8.0)):
                cands.append((i, j, k))
    cands.sort(key=lambda m: -sum(durs[m[0]:m[0] + m[2]]))
    used = [False] * n
    confirmed = []
    for i, j, k in cands:
        if any(used[t] for t in list(range(i, i + k)) + list(range(j, j + k))):
            continue
        pairs_sorted = sorted(((durs[i + t], i + t, j + t) for t in range(k)), reverse=True)
        ok = any(xcorr_best(seg(x, *chunks[ci]), seg(x, *chunks[cj])) > 0.75
                 for _, ci, cj in pairs_sorted[:5])
        if not ok:
            continue
        for t in range(i, i + k):
            used[t] = True
        for t in range(j, j + k):
            used[t] = True
        confirmed.append((i, j, i + k, j + k))
    confirmed.sort()
    return confirmed

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
        total = sum(b - a for a, b in chunks[i1:i2])
        if gap <= MAX_PLAY_GAP and total >= 8.0:
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
        start_c = i1
        for t in range(i1, i2):
            if t in marker_set:
                start_c = t + 1
            if t + 1 < i2 and gap_after(chunks, t) >= 5.0 and t + 1 > start_c:
                start_c = t + 1
        materials.append({"start": chunks[start_c][0], "end": chunks[i2 - 1][1],
                          "plays": 2, "flag": ""})

    first_pair = min((m["start"] for m in materials), default=1e9)

    labels = [m for m in markers if m not in pair_zone and gap_before(chunks, m) >= 4.0]

    candidates = []
    for li, m in enumerate(labels):
        block_end = chunks[labels[li + 1]][0] if li + 1 < len(labels) else 1e9
        s, e, c = chunks[m][1], chunks[m][1], m + 1
        while c < len(chunks) and chunks[c][0] < block_end:
            if c in marker_set:
                break
            if any(chunks[c][0] < p["end"] and chunks[c][1] > p["start"] for p in materials):
                break
            e = chunks[c][1]
            if gap_after(chunks, c) >= 5.0:
                break
            c += 1
        if e - s >= 2.0:
            candidates.append({"start": s, "end": e, "plays": 1, "flag": ""})

    kept = []
    for cd in candidates:
        if cd["end"] <= first_pair:
            kept.append(cd)
        else:
            notes.append(f"dropped candidate @{cd['start']:.1f}-{cd['end']:.1f} "
                         f"({cd['end']-cd['start']:.0f}s, at/after first pair)")
    kept.sort(key=lambda t: t["start"])
    if len(kept) > 5:
        for cd in kept[5:]:
            notes.append(f"dropped candidate @{cd['start']:.1f}-{cd['end']:.1f} "
                         f"({cd['end']-cd['start']:.0f}s, beyond section-1 cap of 5)")
        kept = kept[:5]
    for cd in kept:
        if first_pair - cd["end"] < 30.0:
            cd["flag"] = "CHECK: ends close to first pair"
    materials.extend(kept)

    materials.sort(key=lambda t: t["start"])
    for i, mt in enumerate(materials, 1):
        mt["n"] = i
    return materials, mat_pairs, disc, labels, notes

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

def process(path, outdir):
    name = os.path.splitext(os.path.basename(path))[0]
    x = decode_all(path)
    dur = len(x) / SR
    chunks = build_chunks(detect_silences_np(x), dur)
    markers = find_markers(x, chunks)
    pairs = find_repeat_pairs(x, chunks)
    materials, mat_pairs, disc, labels, notes = extract_materials(x, chunks, markers, pairs)

    print(f"  分析完成: {len(materials)} 段材料 "
          f"({sum(1 for m in materials if m['plays'] == 1)} 段播一遍 + "
          f"{sum(1 for m in materials if m['plays'] == 2)} 段播两遍)")
    for mt in materials:
        print(f"    D{mt['n']:02d}  第{mt['n']}段材料  {mt['start']:7.1f}s - {mt['end']:7.1f}s  "
              f"({mt['end']-mt['start']:.0f}秒, 播{'两' if mt['plays'] == 2 else '一'}遍)")
    if len(materials) != 10:
        print(f"    !! 注意: 找到 {len(materials)} 段 (正常应为 10 段), 请人工核对")

    os.makedirs(outdir, exist_ok=True)
    timeline = {
        "materials": {str(i): {"start": round(mt["start"], 2), "end": round(mt["end"], 2)}
                      for i, mt in enumerate(materials, 1)},
        "questions": {},
        "_meta": {"source": os.path.basename(path), "plays": {
                      str(i): mt["plays"] for i, mt in enumerate(materials, 1)}},
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
        print(f"    !! 第 {bad} 段切分后是静音, 请人工核对")
    if unverified:
        print(f"    ? 第 {unverified} 段音量无法校验 (文件被占用), 可自行抽查")
    if not bad and not unverified:
        print(f"    切分完成, 音量检查全部通过 (最低 {min(vols):.1f}dB)")
    print(f"    结果已保存到: {outdir}")
    return materials

# ---------- main ----------

def pause():
    try:
        input("按回车键退出...")
    except EOFError:
        pass


def main():
    global FF
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    no_pause = "--no-pause" in sys.argv

    print("=" * 46)
    print("  高考英语听力真题 · 自动切分工具")
    print("=" * 46)

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

    print("\n" + "=" * 46)
    print(f"全部完成: 成功 {ok} 个" + (f", 失败 {failed} 个" if failed else ""))
    print("每个 MP3 旁边都有一个「原名-切分结果」文件夹, 里面是 D01-D10。")
    if not no_pause:
        pause()

if __name__ == "__main__":
    main()

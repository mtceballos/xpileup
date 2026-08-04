import numpy as np
from astropy.io import fits


def mean_derivative(derivative, window, offset=1):
    """mean_di[i] = 1/window * sum_{j=1}^{window} derivative[i-j-offset]"""
    n = len(derivative)
    mean_di = np.zeros(n, dtype=float)
    for i in range(window + offset, n):
        mean_di[i] = (1.0 / window) * sum(derivative[i - j - offset] for j in range(1, window + 1))
    return mean_di


def derive_new_kernel(kernel_coeficients, start_sample_for_kernel1, window, offset=1):
    """DDS kernel for dw_i = d_i - mean_derivative(d, window, offset)[i],
    expressed directly in terms of the noise stream s.
    Returns {relative_offset: coefficient} such that
    dw_i = sum(c * s[i + r] for r, c in new_kernel.items())"""
    base = {}
    for k, c in enumerate(kernel_coeficients):
        r = start_sample_for_kernel1 - k
        base[r] = base.get(r, 0.0) + c

    new_kernel = dict(base)  # d_i term
    if window > 0:
        for j in range(1, window + 1):
            shift = -j - offset
            for r, c in base.items():
                m = r + shift
                new_kernel[m] = new_kernel.get(m, 0.0) - c / window

    return {m: c for m, c in new_kernel.items() if c != 0.0}


def derive_new_kernel_array(kernel_coeficients, start_sample_for_kernel1, window, offset, array_length=20):
    """Same as derive_new_kernel, but returned as a coefficient array indexed by
    k = start_sample_for_kernel1 - r, so it can be used directly as
    sum(arr[k] * record_stream[i + start_sample_for_kernel1 - k] for k in range(array_length))"""
    kernel = derive_new_kernel(kernel_coeficients, start_sample_for_kernel1, window, offset)
    arr = np.zeros(array_length, dtype=float)
    for r, c in kernel.items():
        k = start_sample_for_kernel1 - r
        arr[k] = c
    return arr


def derivative_from_kernel(record_stream, kernel_array, start_sample_for_kernel1):
    """Vectorized DDS derivative defined by kernel_array (as produced by
    derive_new_kernel_array), applied to every record in record_stream (n_records, n_samples).
    record_stream must be 2D -- for a single record, pass e.g. all_record_stream[0:1].
    Returns an array of shape (n_records, n_valid_samples)."""
    nz = np.nonzero(kernel_array)[0]
    rs = start_sample_for_kernel1 - nz  # relative offsets for each nonzero tap
    r_min, r_max = rs.min(), rs.max()

    n_samples = record_stream.shape[1]
    j_start = -r_min
    j_end = n_samples - r_max  # exclusive

    deriv = np.zeros((record_stream.shape[0], j_end - j_start), dtype=float)
    for k, r in zip(nz, rs):
        deriv += kernel_array[k] * record_stream[:, j_start + r: j_end + r]

    return deriv


def rms_from_kernel(record_stream, kernel_array, start_sample_for_kernel1):
    """RMS of the derivative_from_kernel derivative, per record.
    Returns an array of shape (n_records,)."""
    deriv = derivative_from_kernel(record_stream, kernel_array, start_sample_for_kernel1)
    return np.sqrt(np.mean(deriv ** 2, axis=1))


def save_fake_detections_fits(filename, rows, positions_list, energies_list):
    """
    rows: list of (record, window, offset, threshold) tuples, one per combination,
          in the same order as positions_list / energies_list.
    positions_list, energies_list: lists of 1D arrays (possibly empty), same length
          as rows -- positions_list[i]/energies_list[i] are the detections for rows[i].
          positions is the fake's arrival sample within its record (TIME rebased to
          the record start and divided by tclock); energies is SIGNAL (keV).

    Note: POSITIONS/ENERGIES are written as plain (uncompressed) ImageHDUs, not
    CompImageHDU -- astropy's tile compression quantizes floating-point data by
    default (lossy, ~0.5 absolute error even with quantize_level=-1), which would
    corrupt arrival samples and energies. If file size becomes a problem, gzip the
    written file afterwards (filename + ".gz") for lossless compression instead;
    fits.open() reads .fits.gz transparently.
    """
    records, windows_col, offsets_col, thresholds_col = (np.asarray(a) for a in zip(*rows))
    n_detections = np.array([len(p) for p in positions_list], dtype=np.int32)

    flat_positions = (np.concatenate(positions_list) if n_detections.sum() else np.array([])).astype(np.float32)
    flat_energies = (np.concatenate(energies_list) if n_detections.sum() else np.array([])).astype(np.float32)

    meta_cols = [
        fits.Column(name="RECORD", format="J", array=records.astype(np.int32)),
        fits.Column(name="WINDOW", format="I", array=windows_col.astype(np.int16)),
        fits.Column(name="OFFSET", format="I", array=offsets_col.astype(np.int16)),
        fits.Column(name="THRESHOLD", format="E", array=thresholds_col.astype(np.float32)),
        fits.Column(name="NDET", format="J", array=n_detections),
    ]
    meta_hdu = fits.BinTableHDU.from_columns(meta_cols, name="METADATA")

    pos_hdu = fits.ImageHDU(data=flat_positions, name="POSITIONS")
    en_hdu = fits.ImageHDU(data=flat_energies, name="ENERGIES")

    fits.HDUList([fits.PrimaryHDU(), meta_hdu, pos_hdu, en_hdu]).writeto(filename, overwrite=True)


def load_fake_detections_fits(filename):
    """
    Returns:
      meta: structured array with fields RECORD, WINDOW, OFFSET, THRESHOLD, NDET
      offsets: 1D int array, len(meta) + 1 -- offsets[i]:offsets[i+1] slices
               positions/energies for row i
      positions, energies: flat 1D arrays
    """
    with fits.open(filename, memmap=True) as hdulist:
        meta = hdulist["METADATA"].data
        positions = hdulist["POSITIONS"].data
        energies = hdulist["ENERGIES"].data

    offsets = np.concatenate(([0], np.cumsum(meta["NDET"])))
    return meta, offsets, positions, energies


def get_detections(offsets, positions, energies, row_index):
    """Positions/energies for a single (record, window, offset, threshold) row."""
    start, end = offsets[row_index], offsets[row_index + 1]
    return positions[start:end], energies[start:end]

from pathlib import Path

import pandas as pd
import pytest
from example_data import LME4_REFERENCE, OUTPUTS

import mera


@pytest.fixture(scope="module")
def results() -> mera.MeraResults:
    """A complete set of saved results, cut to three IMs to keep the round trips quick."""
    full = mera.MeraResults.load(LME4_REFERENCE / "no_mask")
    ims = full.bias_std_df.index
    frames = {name: getattr(full, name) for name in OUTPUTS}
    return mera.MeraResults(
        **{
            # rem_res_df keeps its event and site id columns
            name: frame.loc[ims[:3]]
            if name == "bias_std_df"
            else frame.drop(columns=ims[3:])
            for name, frame in frames.items()
        }
    )


@pytest.mark.parametrize("fmt", ["csv", "parquet"])
def test_results_round_trip(
    tmp_path: Path, fmt: str, results: mera.MeraResults
) -> None:
    if fmt == "csv":
        results.save(tmp_path)
        loaded = mera.MeraResults.load(tmp_path)
    else:
        results.save_to_parquet(tmp_path)
        loaded = mera.MeraResults.load_from_parquet(tmp_path)
    for name in OUTPUTS:
        pd.testing.assert_frame_equal(getattr(loaded, name), getattr(results, name))


def test_fit_and_remaining_residuals_are_optional(
    tmp_path: Path, results: mera.MeraResults
) -> None:
    (tmp_path / "csv").mkdir()
    (tmp_path / "pq").mkdir()
    results.save(tmp_path / "csv", save_fit=False, save_rem=False)
    results.save_to_parquet(tmp_path / "pq", save_fit=False, save_rem=False)
    for loaded in [
        mera.MeraResults.load(tmp_path / "csv"),
        mera.MeraResults.load_from_parquet(tmp_path / "pq"),
    ]:
        assert loaded.fit_df is None and loaded.rem_res_df is None
        pd.testing.assert_frame_equal(loaded.bias_std_df, results.bias_std_df)


def test_progress_and_record_count_warnings(
    example_residuals: pd.DataFrame, capsys: pytest.CaptureFixture[str]
) -> None:
    mera.run_mera(
        example_residuals,
        ["PGA"],
        "event_id",
        "stat_id",
        verbose=True,
        raise_warnings=True,
        min_num_records_per_event=1000,
    )
    out = capsys.readouterr().out
    assert "Processing PGA" in out and "Finished processing PGA" in out
    assert "Warning: For IM PGA" in out


def test_mask_too_few_records_without_initial_mask(
    example_residuals: pd.DataFrame, capsys: pytest.CaptureFixture[str]
) -> None:
    df = example_residuals[["PGA", "pSA_1.0", "event_id", "stat_id"]]
    mask = mera.mask_too_few_records(df, min_num_records_per_event=5)
    assert mask.shape == (len(df), 2)
    assert not mask.to_numpy().all()
    assert "Masked" in capsys.readouterr().out

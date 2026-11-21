import pandas as pd
import pytest

from de_pipeline import quality as dq
from de_pipeline.stages.clean import fix_mojibake, normalise_text, split_title_body, transform


def test_fix_mojibake():
    assert fix_mojibake("costing Â£3.2bn") == "costing £3.2bn"
    assert fix_mojibake("itâ€™s") == "it’s"
    assert fix_mojibake("plain £ text") == "plain £ text"


def test_normalise_and_split():
    text = normalise_text("Title here\n \n First  para.\n \n Second\tpara. ")
    assert text == "Title here\n\nFirst para.\n\nSecond para."
    assert split_title_body(text) == ("Title here", "First para.\n\nSecond para.")


def test_transform_dedupes_filters_and_conforms():
    body = "Body sentence number one. " * 10
    bronze = pd.DataFrame({
        "raw_text": [f"A\n \n {body}", f"A\n \n {body}", "B\n \n tiny", f"C\n \n {body} extra"],
        "raw_label": ["Business", "business", "tech", " TECH "],
        "source": "s", "source_row": range(4), "batch_id": "b",
    })
    out, m = transform(bronze, min_body_chars=50)
    assert m == {"input_rows": 4, "dropped_too_short": 1, "dropped_duplicates": 1, "output_rows": 2}
    assert out["category"].tolist() == ["business", "tech"]
    assert out["doc_id"].is_unique


def test_quality_gate_raises():
    df = pd.DataFrame({"category": ["tech", "cooking"]})
    with pytest.raises(dq.DataQualityError):
        dq.run_checks(df, [dq.accepted_values("category", ["tech"])], "t")
    res = dq.run_checks(df, [dq.accepted_values("category", ["tech"])], "t", fail_on_error=False)
    assert res[0]["passed"] is False

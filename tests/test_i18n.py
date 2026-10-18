from readmission import codes, i18n


def test_both_languages_have_the_same_texts():
    assert set(i18n.TEXT["en"]) == set(i18n.TEXT["es"])
    assert all(i18n.TEXT["es"][key].strip() for key in i18n.TEXT["en"])


def test_every_code_has_a_spanish_name():
    tables = {
        "admission_type": codes.ADMISSION_TYPE,
        "discharge": codes.DISCHARGE_DISPOSITION,
        "source": codes.ADMISSION_SOURCE,
    }
    for which, english in tables.items():
        spanish = i18n.code_labels("es", which)
        assert set(spanish) == set(english)
        assert i18n.code_labels("en", which) == english
        for code, name in english.items():
            if name.startswith("Unknown ("):
                assert spanish[code].startswith("Desconocido (")
            else:
                assert spanish[code] != name  # translated


def test_payers_and_specialties_match_what_the_model_knows(real_like_artifact):
    pipeline = real_like_artifact["pipeline"]
    prep = pipeline.named_steps["prep"]
    known = {}
    for _, transformer, columns in prep.transformers_:
        if hasattr(transformer, "categories_"):
            for column, categories in zip(columns, transformer.categories_, strict=True):
                known[column] = set(categories)
    if "payer_code" in known and len(known["payer_code"]) > 3:  # the real model, not a toy one
        assert set(i18n.PAYERS) <= known["payer_code"]
        assert {value for value, _, _ in i18n.SPECIALTIES} <= known["medical_specialty"]


def test_icd9_shape_and_summary():
    assert i18n.looks_like_icd9("428") and i18n.looks_like_icd9("250.83")
    assert i18n.looks_like_icd9("v45") and i18n.looks_like_icd9("E849")
    assert not i18n.looks_like_icd9("hello") and not i18n.looks_like_icd9("42")
    result = {
        "readmission_probability": 0.152,
        "baseline_risk": 0.112,
        "relative_risk": 1.36,
        "high_risk": False,
        "summary": "english",
    }
    assert i18n.summary("en", result) == "english"
    assert "15 %" in i18n.summary("es", result) or "15%" in i18n.summary("es", result)
    assert "por debajo del umbral de alto riesgo" in i18n.summary("es", result)


def test_spanish_names_work_even_with_the_raw_dataset_labels(monkeypatch):
    raw = {1: "Emergency", 5: "Not Available", 6: "NULL"}
    monkeypatch.setitem(i18n._SPANISH, "admission_type", (raw, i18n._ADMISSION_TYPE_ES))
    labels = i18n.code_labels("es", "admission_type")
    assert labels[1] == "Urgencia"
    assert labels[6] == "Desconocido (figura como 'NULL' en el dataset)"
    assert labels[5] == "Desconocido (figura como 'Not Available' en el dataset)"


def test_every_extra_drug_has_a_spanish_name():
    import ast
    from pathlib import Path

    app = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
    for node in ast.parse(app.read_text()).body:
        if isinstance(node, ast.Assign) and node.targets[0].id == "OTHER_DRUGS":
            assert set(ast.literal_eval(node.value)) == set(i18n.DRUGS_ES)
            return
    raise AssertionError("OTHER_DRUGS not found")

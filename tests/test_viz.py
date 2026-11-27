"""Viewer export (tmc.viz): what the animation shows must match the simulation it was made from."""
import json

import numpy as np
import pandas as pd

from tmc import viz
from tmc.disturbance import load_setting


def test_fallback_runs_merge_consecutive_ticks_per_reason():
    class R:
        cav_events = pd.DataFrame({"time_s": [1.0, 1.1, 1.2, 1.5, 1.6, 2.0], "vehicle_id": [3, 3, 3, 3, 3, 4],
                                   "reason": ["no_message"] * 3 + ["leader_unequipped"] * 2 + ["stale"]})
    runs = viz.fallback_runs(R())
    assert runs[3] == [[1.0, 1.3, "no_message"], [1.5, 1.7, "leader_unequipped"]]
    assert runs[4] == [[2.0, 2.1, "stale"]]


def test_e9a_export_matches_the_run(tmp_path):
    res, name = viz.run_e9a("nominal", "cacc", 0.5, 181)
    d = viz.export(res, follow=0)
    assert d["meta"]["open_road"] and d["meta"]["follow"] == 0
    assert d["meta"]["drivers"] == {"T_r": 0.6, "leader_change": "delayed_gap", "cav": {
        "p": 0.5, "mode": "cacc", "tau_s": 0.1, "delay_s": 0.15, "broadcast": "command", "controller": "eq1126"}}
    s = load_setting()
    assert len(d["veh"]) == int(s["n_followers"]) + 1
    eq = res.vehicles.set_index("vehicle_id")["equipped"]
    for row in d["veh"]:
        assert row[7] == int(eq.loc[row[0]])
        if row[8]:                                        # fallback runs only for equipped vehicles
            assert row[7] == 1
    total = sum(round((b - a) / 0.1) for row in d["veh"] for a, b, _ in row[8])
    assert total == len(res.cav_events)
    assert d["meta"]["L"] < 1e5                           # open road clipped to the recorded extent
    out = viz.write_viewer(res, tmp_path / "v.html", None, 0)
    html = out.read_text(encoding="utf-8")
    assert "/*__DATA__*/null" not in html
    blob = html.split("const D = ", 1)[1].split(";\nconst M = D.meta", 1)[0]
    assert json.loads(blob)["summary"]["n_equipped"] == int(eq.sum())


def test_e9b_human_export_has_no_cav():
    res, name = viz.run_e9b(1.20, "human", 0.0, 181)
    d = viz.export(res)
    assert d["meta"]["drivers"]["cav"] is None
    assert d["meta"]["drivers"]["leader_change"] == "restart" and d["meta"]["dt_s"] == 0.05
    assert all(r[7] == 0 and r[8] == [] for r in d["veh"])
    assert name == "E9B_Xp1.20_human_s181" and not d["meta"]["open_road"]
    assert np.isclose(d["summary"]["total_travel_time_veh_h"], res.summary()["total_travel_time_veh_h"], atol=0.01)


def test_combined_viewer_embeds_each_run_and_decodes_only_the_chosen_one(tmp_path):
    entries = []
    for mode, p in (("human", 0.0), ("cacc", 0.5)):
        res, name = viz.run_e9a("nominal", mode, p, 181)
        entries.append((name, "E9A", mode, viz.export(res, follow=0)))
    html = viz.write_combined(entries, tmp_path / "viewer.html").read_text(encoding="utf-8")
    assert "/*__DATA__*/null" not in html and "/*__RUNS__*/null" not in html and "__RUNDATA__" not in html
    runs = json.loads(html.split("const RUNS = ", 1)[1].split(";\n", 1)[0])
    assert [r["key"] for r in runs] == [e[0] for e in entries]
    for key, _, _, d in entries:                          # each block round-trips to the exported run
        blob = html.split(f'<script type="application/json" id="run-{key}">', 1)[1].split("</script>", 1)[0]
        assert json.loads(blob) == json.loads(json.dumps(d))
    assert 'const D = JSON.parse(document.getElementById("run-" + RUN.key).textContent);' in html
    try:
        viz.write_combined(entries + entries[:1], tmp_path / "dup.html")
        assert False, "duplicate keys accepted"
    except ValueError:
        pass

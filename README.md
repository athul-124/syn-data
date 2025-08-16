# syn-data
# SynData — MVP (Phase A)

**One-liner:** On-demand synthetic tabular data generation for small teams — with automated quality & utility checks.

**Why:** Startups lack data. SynData generates privacy-safe synthetic datasets and proves their value by showing model performance on demo tasks.

**MVP scope (must-haves):**
- CSV upload → generate synthetic tabular CSV (numeric/categorical/datetime).
- Quality report: per-column mean/std + KS-style comparison (simple).
- Demo training: train a basic model on real vs synthetic and show comparison (accuracy/regression MAE).
- API + downloadable CSV.
- Pay-as-you-go placeholder (free credits for pilots).

**Tech stack (MVP):**
- Backend: Python + FastAPI, pandas, numpy, scikit-learn
- Frontend: (later) React / simple static page
- Storage: local filesystem (uploads/outputs)
- Deploy: Render / Railway / Heroku (quick)

**How to run (dev):**
1. `python -m venv venv && source venv/bin/activate`
2. `pip install -r requirements.txt`
3. `uvicorn backend.main:app --reload --port 8000`
4. POST `/generate` with `multipart/form-data` file=`dataset.csv`, json `{"n_rows":1000}`

**Next milestones (3 weeks):**
- Week1: Basic generator + API + demo scripts.
- Week2: Quality report + demo model training + simple UI.
- Week3: Pilot onboarding, small feedback loop, iterate.

**Acceptance criteria (MVP):**
- Endpoint `/generate` returns a synthetically generated CSV for 3 test datasets.
- Quality report contains basic fidelity and utility numbers.
- Demo compares a model trained on real vs synthetic and prints metrics.
- 3 pilot signups and 1 pilot integration scheduled.

**Contact:** YourName — start pilots & testing.

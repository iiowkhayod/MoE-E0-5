# E2 Folsom exploratory protocol

This is a single-seed exploratory campaign on the complete eligible population of the Folsom 2014–2016 release. It is not an official leaderboard protocol. The official CSV and image archives are used without selecting dates or weather classes. Eligibility is determined before training and shared by every condition.

## Split and target

- Train 2014, validate 2015, test 2016.
- Point GHI at +5, +10, +15 and +30 minutes.
- Issue times require 30 complete one-minute GHI observations and six causal images requested at t−25, t−20, t−15, t−10, t−5 and t. The archive modification time is the image clock. The most recent image no later than each request is used, up to 120 seconds old.
- Issue and all target solar zenith angles must be below 80 degrees. This follows the low-sun exclusion used in the Folsom alignment study.
- Clear-sky GHI uses pvlib Ineichen at 38.642°N, 121.148°W, altitude 60 m. No future measured weather enters the clear-sky calculation.

## Conditions

- `smart_persistence`: current clear-sky index times future clear-sky GHI.
- `e1_direct_a4`: same multimodal model and future clear-sky inputs as E2, predicting GHI directly.
- `e2_a1`: historical residual plus physical clear-sky context.
- `e2_a2`: A1 plus conventional weather observations.
- `e2_a3`: A1 plus six historical sky images.
- `e2_a4`: A1 plus weather and images. This is the primary E2 baseline.

All learned conditions use seed 43, GHI-scale MSE, the same batch order policy, and validation MSE for checkpoint selection. Ramp labels never select checkpoints. Ramp is an absolute future GHI change of at least 100 W/m², following the cited DBMF evaluation and the pre-existing Folsom protocol. The outcome is exploratory: report E2−E1 paired metric differences without significance claims.

## Model-family follow-up

After the input ablation, run only three structure alternatives: GRU instead of TCN for A1, a small 3D CNN instead of CNN-GRU for A3, and a single learned fusion gate instead of concatenation for A4. The follow-up retains the full prepared scope.


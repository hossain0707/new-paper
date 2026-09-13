# Checkpoint portability spot check

An initial CPU / 25-row inference check differed from archived GPU inference by up to
0.02171326 mg/dL on the first neural checkpoint. A controlled repeat using the original
GPU math backend and original prediction batch size reproduced the archived TFT
predictions exactly (maximum absolute difference 0.0 mg/dL). The final integrity check
uses the original backend and batches. CPU/GPU bitwise equivalence is not claimed;
no model was retrained and no archived prediction was altered to address this check.

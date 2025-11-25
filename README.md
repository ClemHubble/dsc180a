# Llama-3.1-8B-Instruct Analysis

Welcome to our DSC 180A Capstone Repo! This repo contains code for different methods of analyzing Llama-3.1-8B-Instruct's performance. Each form of analysis is in its own branch.

## QRC
This branch implements a full evaluation pipeline for Quantile Risk Calibration (QRC) on Llama-3.1-8B-Instruct, focusing on how the model’s confidence behaves under different sampling conditions. QRC is a tail-focused calibration metric that examines model performance on the top-k% most confident predictions—in our case, the top 10%. Unlike traditional accuracy or ECE, which average over all predictions, QRC isolates the model’s strongest claims and tests whether high confidence actually corresponds to high correctness.  

Large language models are increasingly deployed in high-risk settings where overconfidence can be dangerous. QRC addresses this by focusing specifically on the region where the model expresses the most confidence. A model with high accuracy but poor QRC is unreliable when it matters most; a model with strong QRC provides meaningful confidence signals that downstream systems (and humans) can trust.

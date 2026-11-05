# Hardware for the current catalog

Numbers below were measured on this machine while `python -m knowledge_graph.cli ingest` was building the graph for `knowledge_graph/raw_data/own_products/products_data.csv`. The graph backend in that run is the in-memory store, not Neo4j. OpenRouter runs in the cloud, so the chatbot does not need a local GPU.

## What is stored

| Item | Size |
| --- | --- |
| Product rows | 124,682 |
| Raw CSV | 39.9 MB (41,855,107 bytes) |
| Envelope files | 124,688 files, 715.7 MB (750,439,130 bytes) |
| Average envelope | 6,019 bytes |

Envelopes are written under `knowledge_graph/generated/envelopes/own_products/`. One ingest keeps every envelope in memory and then builds the product graph in the same process.

## Memory observed during the build

The ingest process (PID 4688) was using **4,119 MB** working set and **4,123 MB** private memory after the log line `updated=124682`, while the graph build was still running.

That is the Python process only. Windows and the conda environment need memory on top of it.

## Machine to use

| Resource | For this catalog |
| --- | --- |
| RAM | 16 GB. The build already uses about 4.1 GB, and the graph is still growing in that same process. 8 GB is too small for this run plus the operating system. |
| Free disk | 4 GB. The envelopes alone are 716 MB. Leave room for ingestion state and the graph snapshot written at the end of ingest. |
| CPU | 4 cores. Ingest and the graph build run as one Python process. Extra cores do not split this job. |
| GPU | None. Answers go to OpenRouter over the network. |

A laptop with 16 GB RAM and a normal SSD can run this catalog. Keep other large applications closed while ingest is running.

## What gets heavier

A second ingest of an unchanged CSV rewrites the same envelopes and builds the graph again. Plan on the same 4 GB-plus process, not a small incremental update.

Switching `FASHIONGPT_GRAPH_BACKEND` to Neo4j adds a database process. The memory graph measured here does not include that. Keep Neo4j off unless a separate database machine or extra RAM is available.

The raw CSV is 40 MB. Copying it, or the 716 MB envelope folder, needs that much free space on the destination disk.

## If the model runs on Ollama

OpenRouter does not use this PC for the model. Ollama does. The catalog process and the model are both resident while Fashion Brain answers.

This PC has an NVIDIA GeForce RTX 4090 with 23,028 MiB of VRAM. At the time of this note, 22,446 MiB of that VRAM was already in use, so Ollama cannot place a model on the GPU until that memory is free. If the GPU stays full, Ollama uses system RAM instead.

| Model size (4-bit) | Extra memory for the model | Where it fits |
| --- | --- | --- |
| About 7B to 8B | 6 to 8 GB | RTX 4090 when VRAM is free, or system RAM |
| About 14B | 10 to 12 GB | RTX 4090 when VRAM is free |
| About 32B | about 20 GB | RTX 4090 only when almost all 24 GB is free. Too large to add beside the catalog in 16 GB of system RAM |

Add that model memory to the catalog process, which was already about 4.1 GB during the graph build.

| How you run it | RAM |
| --- | --- |
| 8B model on a free GPU, catalog in system RAM | 16 GB system RAM, and at least 8 GB of free VRAM |
| 8B model on CPU because the GPU is busy | 32 GB system RAM. 16 GB is too small for the 4.1 GB catalog process plus an 8B model |
| 32B model on the RTX 4090 | 24 GB VRAM free before Ollama starts, plus 16 GB system RAM for the catalog |

Disk grows by the model file as well: about 5 GB for an 8B model and about 20 GB for a 32B model. Keep that space free in addition to the 4 GB used for envelopes and the graph snapshot.

Ollama listens on this machine, usually at `http://127.0.0.1:11434`. No OpenRouter key is required. Ingest can still run without Ollama. Start Ollama only when you ask questions, and do not start it while the GPU is already full.

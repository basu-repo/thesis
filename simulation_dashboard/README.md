# Thesis Simulation Dashboard

This lightweight local web app lets you:

- launch the `07` rule-based baseline or the `09` learned-model live evaluation
- check or uncheck major runtime features before launch
- watch live console output in the browser
- monitor process status, PID, start time, elapsed time, and generated resource-log paths
- keep a GZweb viewer URL in the same page for browser-based Gazebo viewing

## Run

Use the Thesis Python environment if available:

```bash
/home/basudeo/miniconda3/envs/tct/bin/python /home/basudeo/Documents/Thesis/simulation_dashboard/dashboard_server.py
```

Or use the current interpreter:

```bash
python3 /home/basudeo/Documents/Thesis/simulation_dashboard/dashboard_server.py
```

Then open:

```text
http://127.0.0.1:8765
```

## What the dashboard controls

### 07 Rule-Based Baseline

- headless Gazebo
- RViz on or off
- camera viewer on or off
- bag recording on or off
- depth classification on or off
- hazard map on or off
- decision fuser on or off
- UAV scouts on or off
- second UAV on or off
- lidar straight-approach on or off
- lidar path-planning on or off
- debug isolate mode

### 09 Learned-Model Evaluation

- model selection or explicit checkpoint
- headless Gazebo
- RViz on or off
- camera viewer on or off
- target index
- OMNeT communication co-simulation on or off
- OMNeT configuration selection

## GZweb

The dashboard does not install or build GZweb automatically. It provides:

- a stored GZweb URL field
- an iframe area for the browser viewer
- a direct link to the install guide

Guide used:

- https://github.com/Intelligent-Quads/iq_tutorials/blob/master/docs/gzweb_install.md

Important note:

- the current Thesis simulation stacks use `ign gazebo` / Gazebo Sim
- the linked GZweb guide targets the older Gazebo Classic toolchain
- so the dashboard is ready now, but the browser-world viewer should be treated as a separate compatibility step

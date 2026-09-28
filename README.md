# Co-π-tree: An Interpretable Policy Tree for Human-AI Collaboration

This repository contains the Overcooked-AI implementation for
[*Distilling LLM Reasoning into an Interpretable Policy Tree for Human-AI Collaboration*](https://arxiv.org/pdf/2606.08596).

Co-π-tree (Collaboration Policy Tree) learns an executable collaboration policy.
One tree predicts a teammate's behavior; a second tree uses that prediction to
select the controlled agent's action. The command-line implementation keeps the
historical name <code>llmtree</code> for Co-π-tree.

During training, an LLM constructs the policy tree, generates code, and revises
local branches from environment feedback. Evaluation directly executes the final
Python policy code and does not call an LLM at every step. The paper studies
collaboration with both AI and human teammates.

## Main experiment: Co-π-tree with unseen AI teammates

The AI teammate pool is <code>SP</code> (Self-Play), <code>PBT</code>,
<code>FCP</code>, <code>MEP</code>, <code>COLE</code>, and <code>BC</code>.
For each layout and player position, one source teammate is used to construct
and revise a Co-π-tree policy. That policy is evaluated with the other five,
unseen teammates. Co-π-tree is always the method being evaluated.

For example, a policy constructed with SP is tested with PBT, FCP, MEP, COLE,
and BC. The experiment repeats this procedure for every source teammate and
aggregates the results. Each combination is evaluated for five 400-step
episodes, with Co-π-tree controlling Player 0 and Player 1 in separate runs.
See the [paper](https://arxiv.org/pdf/2606.08596) for the complete protocol.

Table 1 additionally compares aggregated Co-π-tree results with SP, PBT, FCP,
MEP, COLE, BC, ProAgent, and CausalPlan. This is separate from the source and
unseen-teammate protocol above.

| Layout in the paper | Command-line name | Model directory |
| --- | --- | --- |
| Cramped Room | <code>cramped_room</code> | <code>simple</code> |
| Coordination Ring | <code>coordination_ring</code> | <code>random1</code> |
| Counter Circuit | <code>counter_circuit</code> | <code>random3</code> |
| Asymmetric Advantages | <code>asymmetric_advantages</code> | <code>unident_s</code> |
| Forced Coordination | <code>forced_coordination</code> | <code>random0</code> |

## Repository layout

~~~text
co-π-Tree/
├── README.md                        # Project documentation
├── LICENSE                          # Project license
├── THIRD_PARTY_NOTICES.md           # Third-party sources and licenses
├── environment.yml                  # Conda environment
├── requirements.txt                 # Python dependencies
├── lib/                             # Locally modified runtime libraries
├── human_ai_demo/                   # Browser-based human-AI experiment interface
├── src1/
│   ├── main.py                      # Policy construction and evaluation entry point
│   ├── api/                         # Planner, coder, and summarizer
│   ├── LLM/                         # LLM API calls
│   ├── prompts/                     # Prompts for tree construction and revision
│   ├── policy/                      # Policy code by layout and player position
│   ├── test_Overcooked/             # Evaluation, policy execution, and teammate adapters
│   └── models/                      # Downloaded AI teammate models; not tracked in Git
└── results/                         # Generated experiment outputs; not tracked in Git
~~~

<code>src1/policy/&lt;layout&gt;_0.txt</code> and
<code>&lt;layout&gt;_1.txt</code> are used when Co-π-tree controls Player 0
and Player 1, respectively. A policy file defines
<code>PredictTeammateThenPlan(state)</code>, which returns the predicted
teammate skill and the controlled agent's selected skill. The executor in
<code>src1/test_Overcooked/llmtree/llmtree_agent.py</code> converts skills
into environment actions.

### The role of <code>lib/</code>

<code>lib/</code> contains the locally modified Overcooked-AI and
Stable-Baselines libraries required by this project. Changes include project
layouts, interface compatibility, reachable counter goals for Co-π-tree, and
priority for ready pots. Install these copies; do not replace them with
packages of the same name from another source.

<code>lib/</code> is not the code directory for the ProAgent or CausalPlan
baselines. The ProAgent adapter is in
<code>src1/test_Overcooked/proagent/</code>. This repository does not include
an executable CausalPlan implementation. ProAgent and CausalPlan data in
<code>src1/plot_data/</code> are only used to draw comparison figures. See
[lib/README.md](lib/README.md) for details.

## 1. Set up the environment

From the repository root:

~~~bash
conda env create -f environment.yml
conda activate co-pi-Tree
pip install -r requirements.txt
pip install -e ./lib/overcooked_ai
pip install -e ./lib/stable-baselines
~~~

The environment uses Python 3.7 and TensorFlow 1.x to load legacy baseline
models. Check that the packages resolve to this repository's <code>lib/</code>:

~~~bash
python src1/main.py --help
python -c "import sys, importlib.util; sys.path.insert(0, 'src1'); import project_paths; print(importlib.util.find_spec('overcooked_ai_py').origin); print(importlib.util.find_spec('stable_baselines').origin)"
~~~

Check the policy executor's counter-pickup and ready-pot-priority logic:

~~~bash
python -m unittest discover -s tests -p 'test_goal_preferences.py'
~~~

## 2. Download AI teammate models

SP, PBT, FCP, MEP, COLE, and BC require pretrained models. Download them from
the [COLE-Platform Google Drive folder](https://drive.google.com/drive/folders/1s88a_muyG6pVlfcKDKop6R1Fhxr8dcGH)
linked by the [official ProAgent setup instructions](https://github.com/PKU-Alignment/ProAgent#how-to-setup).

Although ProAgent uses <code>./models</code>, this repository reads
<code>src1/models/</code>. Put the layout directories and <code>bc_runs</code>
directly there, without another nested <code>models/</code> directory.

~~~text
src1/models/
├── simple/
│   ├── SP/
│   │   ├── params.txt
│   │   ├── saved_model.pb
│   │   └── variables/
│   ├── PBT/
│   ├── FCP/
│   ├── MEP/
│   └── COLE/
├── random0/
├── random1/
├── random3/
├── unident_s/
└── bc_runs/
    └── simple_bc_test_seed0/
        ├── bc_metadata.pickle
        └── model.pkl
~~~

Every layout needs its corresponding
<code>&lt;layout&gt;/&lt;algorithm&gt;/params.txt</code> and model files, plus
<code>bc_runs/&lt;layout&gt;_bc_test_seed0/</code>. For example:

~~~bash
test -f src1/models/simple/SP/params.txt \
  && test -f src1/models/simple/SP/saved_model.pb \
  && echo "SP model files are ready"
~~~

<code>Greedy</code>, <code>Random</code>, and <code>Stay</code> need no
pretrained models.

## 3. Main program workflows

<code>src1/main.py</code> supports three workflows:

| Workflow | Purpose | Main arguments |
| --- | --- | --- |
| Policy construction and revision | Build Co-π-tree with one source teammate | <code>--mode train</code> |
| Unseen-teammate evaluation | Exclude the source teammate and test the other five | <code>--mode test --plan_path ... --test_alg ...</code> |
| Co-π-tree versus Co-π-tree | Inspect two specified policies | <code>--mode test --p0 llmtree --p1 llmtree</code> |

Construct a policy:

~~~bash
python src1/main.py \
  --layout cramped_room \
  --p0 llmtree --p1 SP \
  --mode train \
  --horizon 400
~~~

Evaluate a policy with unseen teammates:

~~~bash
python src1/main.py \
  --layout cramped_room \
  --p0 llmtree --p1 SP \
  --mode test \
  --plan_path results/main_runs/<experiment-directory>/plan_XX.json \
  --test_alg SP \
  --horizon 400 --test_times 5
~~~

Run two Co-π-tree policies together:

~~~bash
python src1/main.py \
  --layout cramped_room \
  --p0 llmtree --p1 llmtree \
  --mode test \
  --p0_policy_path src1/policy/cramped_room_0.txt \
  --p1_policy_path src1/policy/cramped_room_1.txt \
  --horizon 400 --test_times 5
~~~

Exactly one player must be <code>llmtree</code> during policy construction and
unseen-teammate evaluation. Swap <code>llmtree</code> with the source teammate
to construct or evaluate a Player 1 policy. <code>llmtree</code> is the
historical internal command-line name for Co-π-tree.

## 4. Evaluate existing Co-π-tree policies directly

<code>src1/policy/</code> contains policies for five layouts and both player
positions. This command evaluates **Co-π-tree as Player 0 with SP as Player 1**
for five 400-step episodes. It does not call an LLM, but requires the SP model:

~~~bash
python -c "import sys; from types import SimpleNamespace; sys.path.insert(0, 'src1'); from test_Overcooked.test_code_multi import test; args=SimpleNamespace(layout='cramped_room', p0='llmtree', p1='SP', horizon=400, test_times=5, save_replay=False); print(test(args)['result'])"
~~~

For the Player 1 policy, use <code>p0='SP', p1='llmtree'</code>. Replace SP
with PBT, FCP, MEP, COLE, or BC after downloading the corresponding model.

## 5. Generate and revise policies with an LLM

LLM settings are in <code>src1/configs/llm_api_config.py</code>. By default,
the planner, coder, and summarizer use <code>gpt-4o</code>. Before training,
configure an API key and compatible endpoint:

~~~bash
export CO_PI_TREE_GPT4O_API_KEY=your_api_key
export CO_PI_TREE_GPT4O_BASE_URL=your_api_endpoint
~~~

<code>main.py</code> currently runs ten policy iterations. Results are written
to
<code>results/main_runs/exp_&lt;timestamp&gt;_layout_&lt;layout&gt;_p0-&lt;algorithm&gt;_p1-&lt;algorithm&gt;/</code>.
Each iteration stores <code>plan_00.json</code> through
<code>plan_09.json</code>. Their <code>final_code</code> fields contain the
generated policy code, and LLM usage records are saved alongside them.

In <code>--mode test</code>, <code>--test_alg</code> identifies the source
teammate used during construction; the command automatically evaluates the
other five. Replace <code>&lt;experiment-directory&gt;</code> and
<code>plan_XX.json</code> with actual outputs. Use
<code>find results/main_runs -name 'plan_*.json'</code> to locate policies.
Test outputs are saved under <code>results/main_runs/test_...</code>.

## Other tools

- <code>--save_replay --replay_gif</code> exports a replay; <code>--replay_dir</code> chooses its output directory.
- <code>src1/layout_transfer_experiment.py</code> and <code>src1/plot_layout_transfer_heatmap.py</code> support cross-layout transfer analysis. Replace angle-bracket paths in <code>src1/layout_transfer_config_template.json</code> with actual training outputs.
- <code>src1/picture.py</code>, <code>src1/picture_tokens.py</code>, and <code>src1/box_*.py</code> generate figures and may require separately prepared results or data.
- A <code>Greedy</code> or <code>Stay</code> game provides a quick environment and policy-execution check.

## Human-AI experiment

The paper's human-AI experiment uses a browser interface. Its code and
standalone instructions are in [human_ai_demo/](human_ai_demo/README.md). It
uses this repository's modified <code>lib/overcooked_ai/</code>, Co-π-tree
executor, and <code>src1/policy/</code>; it does not depend on the ProAgent or
CausalPlan repositories.

~~~bash
cd human_ai_demo
./up.sh
# Open http://localhost:8080 in a browser
~~~

When data collection is enabled, trajectories are written to
<code>human_ai_demo/data/</code>, which is not tracked by Git.
<code>query_count</code> records LLM request count and <code>total_tokens</code>
records token usage from API responses.

## Policy-file safety

The policy executor runs <code>final_code</code> from
<code>plan_XX.json</code> and <code>.txt</code> policy files as Python code.
Only run policies that you generated or reviewed.

## Sources and paper

Co-π-tree builds on the Overcooked-AI evaluation interfaces and baseline
conventions from [ProAgent](https://github.com/PKU-Alignment/ProAgent). The
environment is based on [Overcooked-AI](https://github.com/HumanCompatibleAI/overcooked_ai).
See the [paper](https://arxiv.org/pdf/2606.08596) for the complete method and
experimental definition.

Project code is released under the [MIT License](LICENSE). See
[third-party notices](THIRD_PARTY_NOTICES.md) for the provenance of third-party
code and pretrained models.

# Collaboration-Policy-Tree

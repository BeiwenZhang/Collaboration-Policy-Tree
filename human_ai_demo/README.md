# Co-π-tree Human-AI Experiment Interface

This directory provides the browser-based Overcooked interface used for the
paper's human-AI experiments. The interface is adapted from the
[Overcooked-AI](https://github.com/HumanCompatibleAI/overcooked_ai) demo and
directly uses this repository's:

- <code>../lib/overcooked_ai/</code>: the locally modified Overcooked runtime;
- <code>../src1/test_Overcooked/llmtree/</code>: the Co-π-tree executor;
- <code>../src1/policy/</code>: policy files for five layouts and both player positions.

It does not need a copy of the ProAgent or CausalPlan repositories, nor
pretrained AI-baseline models.

## Start the interface

Install Docker and Docker Compose, then run:

~~~bash
cd human_ai_demo
./up.sh
~~~

Open <http://localhost:8080> in a browser. To stop the service:

~~~bash
cd human_ai_demo
./down.sh
~~~

Bootstrap, Popper, Socket.IO, and Phaser are included at fixed versions in
<code>server/static/vendor/</code>, so the browser does not need external CDN
access after the service starts.

If port 8080 is occupied:

~~~bash
HOST_PORT=8081 ./up.sh
~~~

The interface listens only on the local machine by default. To explicitly allow
external access on an experiment server:

~~~bash
BIND_ADDRESS=0.0.0.0 HOST_PORT=8080 ./up.sh
~~~

To inspect service status and logs:

~~~bash
docker compose ps
docker compose logs -f app
~~~

## Run a human-AI game

On the home page, set one player to <code>Human Keyboard Input</code> and the
other to <code>Co-π-tree</code>, choose a layout, and create a game. The human
player uses the arrow keys to move and the space bar to interact. Co-π-tree can
be Player 1 or Player 2; the server automatically selects the corresponding
<code>&lt;layout&gt;_0.txt</code> or <code>&lt;layout&gt;_1.txt</code> policy.

<code>server/config.json</code> controls available layouts, game duration,
frame rate, and the experiment order at <code>/predefined</code>. Run
<code>./up.sh</code> again after changing this configuration.

The paper's human-AI protocol asks participants to complete at least five
practice rounds. In the formal evaluation, every method is tested with the AI
as Player 1 and Player 2, with method order randomized within each layout. The
selection interface currently includes the full Co-π-tree policy. The policy
artifacts for the <code>Co-π-tree-PI</code> and <code>Co-π-tree-w/o P</code>
ablations are not yet included.

## Data and privacy

When <code>Data Collection</code> is enabled, trajectories are written to
<code>human_ai_demo/data/</code>. Git ignores this directory to prevent
participant trajectories from being accidentally committed. A trajectory
contains game states, actions, timestamps, and a server-generated connection
identifier. Before collecting data, explain its contents, use, retention period,
and deletion procedure in the approved consent materials.

Removing containers does not delete results under <code>data/</code>. Archive
de-identified results separately; do not commit raw trajectories.

## Scope and model requirements

The public interface exposes only <code>Human</code> and <code>Co-π-tree</code>
to keep the human-AI experiment lightweight and reproducible. Command-line
experiments with SP, PBT, FCP, MEP, COLE, and BC still require models under
<code>src1/models/</code>, as described in the root README; they are not
dependencies of this Web interface.


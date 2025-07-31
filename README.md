# GenRL: Building Flexible, Decentralized Multi-Agent RL Environments
<p align="center">
  🌐 <a href="https://www.gensyn.ai/"><strong>Visit Website</strong></a> &nbsp;&nbsp;
  🧠 <a href="https://github.com/gensyn-ai/rl-swarm/blob/main/README.md"><strong>RL Swarm</strong></a> &nbsp;&nbsp;
  ✖️ <a href="https://x.com/gensynai"><strong>X</strong></a> &nbsp;&nbsp;
  🤗 <a href="https://huggingface.co/Gensyn"><strong>Hugging Face</strong></a> &nbsp;&nbsp;
  💬 <a href="https://discord.com/invite/bgyDTy39jZ"><strong>Discord</strong></a>
</p>

<p align="center">
  🔬 <a href="https://www.gensyn.ai/research"><strong>Research</strong></a> &nbsp;&nbsp;
  📰 <a href="https://www.gensyn.ai/resources"><strong>Latest News</strong></a> &nbsp;&nbsp;
  💼 <a href="https://www.gensyn.ai/careers"><strong>Work for Gensyn</strong></a> &nbsp;&nbsp;
  📊 <a href="https://dashboard.gensyn.ai/"><strong>Dashboard</strong></a>
</p>

GenRL is a framework that provides native support for horizontally scalable, multi-agent, multi-stage RL with decentralized coordination and communication.

## Customizable Components:
- **DataManager**: Specifies and manages the particular data your RL environment will use. This could be a text dataset, an image dataset, a chessboard, or something else entirely.
- **RewardManager**: This is where you implement your custom reward functions, directly shaping the RL objective for your agents.
- **Trainer**: Performs two functions
    - *Train*: Manages the core learning process itself, this is where policy updates happen.  Whether you're working with policy gradient optimization, value function approximation, or other RL paradigms, the algorithmic policy updates take place here.
    - *Generation*: Handles the generation of rollouts and agent interactions within the environment.

## Core Components

- **GameManager**: Seamlessly coordinates the data flow between the core components you define and the other agents in the multi-agent swarm.
- **CommunicationManager**: Handles the communication between the agents in the swarm. Current backends include
    - *HiveMind*: A decentralized communication protocol that allows agents to communicate with each other.
    - *Torch Distributed*: A distributed training protocol that allows agents to train with each other.

## Optional Components

- **Coordination**: Handles coordination and orchestration between agents in a decentralized swarm.  This is implemented using smart contracts on the blockchain and is only required when running in a decentralized swarm.

## Framework Defined Progression

We track the progression of the game on a per-round basis.  Each round the data manager initializes round data.  The round data kicks off the game’s stages, for each stage rollouts are generated, appended to the game state, and communicated to the swarm.  After the agent has progressed through the game’s predefined stages, rewards are evaluated and policies are updated. The user has full control over the update, which occurs in the Trainer.train method, and so has the opportunity to update the policy on a per stage or per round basis. 
![orchestrated data flow through the framework](assets/data-flow.png)

## Example Usage
```bash
pip install .[examples]
export NUM_PROC_PER_NODE=1
export NUM_NODES=1
export MASTER_ADDR="localhost"
export MASTER_PORT=29500
./scripts/train.sh $NUM_NODES $NUM_PROC_PER_NODE multistage_math msm_dapodata_grpo.yaml
```

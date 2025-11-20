import torch
from game.snake_game import SnakeGameAI
from agent.dqn_agent import DQNAgent
from visualization import plotter
from visualization.heatmap import show_q_heatmap

def train():
    scores = []
    mean_scores = []
    total_score = 0
    record = 0
    loss_history = []
    games = 0
    # Rainbow DQN: Double DQN + Dueling + Prioritized Replay + longevity rewards
    agent = DQNAgent(use_double_dqn=True, use_dueling=True, use_prioritized_replay=True)
    game = SnakeGameAI()

    try:
        while game.running:
            game.process_events()

            if game.save_model:
                agent.model.save('saved_model.pth')
                print('Model saved as saved_model.pth')
                game.save_model = False
                game.save_feedback = 90  # ~1.5 seconds visual feedback

            if game.load_model:
                try:
                    agent.model.load_state_dict(torch.load('saved_model.pth'))
                    print('Model loaded from saved_model.pth')
                    game.load_feedback = 90
                except FileNotFoundError:
                    print('Saved model not found')
                game.load_model = False

            if game.show_heatmap:
                show_q_heatmap(agent, game.grid_size, 0)
                game.show_heatmap = False

            if game.export_onnx:
                success = agent.export_to_onnx('model.onnx')
                if success:
                    print('ONNX export successful: model.onnx')
                    game.onnx_feedback = 90
                else:
                    print('ONNX export failed')
                game.export_onnx = False

            if game.paused:
                game.render()
                continue

            # Get old state
            state_old = agent.get_state(game)

            # Get move
            final_move = agent.get_action(state_old)

            # Perform move and get new state
            state, reward, game_over = game.step(final_move)
            state_new = agent.get_state(game)
            score = game.score

            # Train short memory
            agent.train_short_memory(state_old, final_move, reward, state_new, game_over)

            # Remember
            agent.remember(state_old, final_move, reward, state_new, game_over)

            if game_over:
                # Train long memory, plot result
                loss = agent.train_long_memory()
                if loss is not None:
                    loss_history.append(loss)
                game.reset()
                agent.decay_epsilon()

                if score > record:
                    record = score
                    agent.model.save('best_model.pth')

                scores.append(score)
                total_score += score
                game.games_count = len(scores)
                game.epsilon_value = agent.epsilon
                mean_score = total_score / len(scores)
                mean_scores.append(mean_score)
                plotter.plot(scores, mean_scores, loss_history)
                print(f'Game {len(scores)}, Score: {score}, Record: {record}, Mean Score: {mean_score:.2f}')
    except Exception as e:
        print(f'Error during training: {e}')

if __name__ == '__main__':
    train()

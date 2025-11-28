"""
Live plotting module for training visualization.
Separated from main training loop to avoid blocking issues.

Key features:
- Shows "Loading..." state until data arrives
- Robust error handling to prevent crashes
- Non-blocking updates
"""

import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for better Windows compatibility
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import threading
import time
import queue


class LivePlotter:
    """Non-blocking live plotter for training metrics with loading state."""
    
    def __init__(self, max_episodes=1000):
        """
        Initialize the live plotter with a loading screen.
        
        Args:
            max_episodes: Maximum number of episodes to display at once
        """
        self.max_episodes = max_episodes
        self.episode_rewards = deque(maxlen=max_episodes)
        self.avg_rewards = deque(maxlen=max_episodes)
        self.episodes = deque(maxlen=max_episodes)
        self.best_avg_reward = float('-inf')
        self.num_timesteps = 0
        
        # Thread-safe lock for updating data
        self.lock = threading.Lock()
        
        # Flag to track if we have data
        self.has_data = False
        
        # Create figure and axes
        try:
            self.fig, self.axes = plt.subplots(2, 2, figsize=(14, 9))
            self.fig.suptitle('PPO Training Progress', fontsize=16, fontweight='bold')
            
            # Show loading screen initially
            self._show_loading_screen()
            
            # Show the plot window
            plt.ion()
            plt.tight_layout()
            self.fig.canvas.draw()
            plt.show(block=False)
            
            # Force a window update
            self.fig.canvas.flush_events()
            
        except Exception as e:
            print(f"Warning: Could not create plot window: {e}")
            self.fig = None
            self.axes = None
        
        self.is_closed = False
        
        # Connect close event
        if self.fig is not None:
            self.fig.canvas.mpl_connect('close_event', self._on_close)
    
    def _show_loading_screen(self):
        """Display a loading message on all plots."""
        if self.axes is None:
            return
            
        for i, ax in enumerate(self.axes.flat):
            ax.clear()
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            
            if i == 0:
                ax.text(0.5, 0.5, 'LOADING...\n\nWaiting for training data\n\nThis may take a moment',
                       ha='center', va='center', fontsize=14,
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
            elif i == 1:
                ax.text(0.5, 0.5, 'Training in progress...\n\nRewards will appear here\nonce episodes complete',
                       ha='center', va='center', fontsize=11,
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
            elif i == 2:
                ax.text(0.5, 0.5, 'Recent performance\nwill show here',
                       ha='center', va='center', fontsize=11,
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
            elif i == 3:
                ax.text(0.5, 0.5, 'Statistics\nwill show here',
                       ha='center', va='center', fontsize=11,
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8))
    
    def _setup_axes(self):
        """Setup axis labels and styling."""
        if self.axes is None:
            return
            
        # Plot 1: Episode Rewards
        self.axes[0, 0].set_xlabel('Episode', fontsize=10)
        self.axes[0, 0].set_ylabel('Total Reward', fontsize=10)
        self.axes[0, 0].set_title('Episode Rewards Over Time', fontsize=11, fontweight='bold')
        self.axes[0, 0].grid(True, alpha=0.3, linestyle='--')
        
        # Plot 2: Moving Average Trend
        self.axes[0, 1].set_xlabel('Episode', fontsize=10)
        self.axes[0, 1].set_ylabel('Avg Reward (50eps)', fontsize=10)
        self.axes[0, 1].set_title('Training Progress (50-episode average)', fontsize=11, fontweight='bold')
        self.axes[0, 1].grid(True, alpha=0.3, linestyle='--')
        
        # Plot 3: Recent Performance
        self.axes[1, 0].set_xlabel('Recent Episodes', fontsize=10)
        self.axes[1, 0].set_ylabel('Reward', fontsize=10)
        self.axes[1, 0].set_title('Last 100 Episodes', fontsize=11, fontweight='bold')
        self.axes[1, 0].grid(True, alpha=0.3, linestyle='--')
        
        # Plot 4: Statistics (text only)
        self.axes[1, 1].axis('off')
    
    def _on_close(self, event):
        """Handle plot window close event."""
        self.is_closed = True
    
    def update(self, episode_reward, avg_reward, num_timesteps, best_avg_reward=None):
        """
        Update the plot with new data.
        
        Args:
            episode_reward: Reward for the latest episode
            avg_reward: Moving average reward
            num_timesteps: Total training timesteps
            best_avg_reward: Best average reward seen so far
        """
        if self.is_closed or self.fig is None:
            return False
        
        with self.lock:
            self.episode_rewards.append(episode_reward)
            self.avg_rewards.append(avg_reward)
            self.episodes.append(len(self.episode_rewards))
            self.num_timesteps = num_timesteps
            self.has_data = True
            
            if best_avg_reward is not None:
                self.best_avg_reward = best_avg_reward
        
        # Update plots (non-blocking)
        try:
            self._redraw()
            return True
        except Exception as e:
            # Don't crash training if plotting fails
            return False
    
    def _redraw(self):
        """Redraw all plots with current data."""
        if self.fig is None or self.axes is None:
            return
            
        if not self.has_data or len(self.episode_rewards) == 0:
            return
        
        with self.lock:
            episodes = list(self.episodes)
            ep_rewards = list(self.episode_rewards)
            avg_rewards = list(self.avg_rewards)
            best_reward = self.best_avg_reward
            timesteps = self.num_timesteps
        
        try:
            # Clear all axes
            for ax in self.axes.flat:
                ax.clear()
            
            # Re-setup axes
            self._setup_axes()
            
            # Plot 1: Episode Rewards with moving average
            self.axes[0, 0].plot(episodes, ep_rewards, 'b-', alpha=0.3, linewidth=0.8, label='Episode Reward')
            if len(avg_rewards) > 0:
                self.axes[0, 0].plot(episodes, avg_rewards, 'r-', linewidth=2.5, label='Avg (50eps)', zorder=5)
            self.axes[0, 0].axhline(y=0, color='g', linestyle=':', alpha=0.5, linewidth=1.5)
            self.axes[0, 0].legend(loc='upper left', fontsize=9)
            
            # Plot 2: Moving Average Trend
            if len(avg_rewards) > 5:
                self.axes[0, 1].plot(episodes, avg_rewards, 'r-', linewidth=2.5)
                if best_reward != float('-inf'):
                    self.axes[0, 1].axhline(y=best_reward, color='g', linestyle='--', 
                                           linewidth=2, label=f'Best: {best_reward:.1f}')
                self.axes[0, 1].axhline(y=0, color='gray', linestyle=':', alpha=0.5, linewidth=1.5)
                self.axes[0, 1].legend(loc='upper left', fontsize=9)
            else:
                self.axes[0, 1].text(0.5, 0.5, 'Waiting for more data...',
                                    ha='center', va='center', fontsize=12)
            
            # Plot 3: Recent Performance (last 100 episodes)
            if len(ep_rewards) > 5:
                recent_count = min(100, len(ep_rewards))
                recent_rewards = ep_rewards[-recent_count:]
                recent_eps = list(range(len(recent_rewards)))
                recent_mean = np.mean(recent_rewards)
                
                self.axes[1, 0].plot(recent_eps, recent_rewards, 'b-', alpha=0.6, linewidth=1.2)
                self.axes[1, 0].axhline(y=recent_mean, color='r', linestyle='--', 
                                       linewidth=2, label=f'Mean: {recent_mean:.1f}')
                self.axes[1, 0].legend(loc='upper left', fontsize=9)
            else:
                self.axes[1, 0].text(0.5, 0.5, 'Waiting for more episodes...',
                                    ha='center', va='center', fontsize=12)
            
            # Plot 4: Training Statistics
            current_avg = avg_rewards[-1] if avg_rewards else 0
            recent_avg = np.mean(ep_rewards[-20:]) if len(ep_rewards) >= 20 else np.mean(ep_rewards) if ep_rewards else 0
            
            # Determine training status
            if len(ep_rewards) < 10:
                status = "🔄 Warming up..."
            elif current_avg > 0:
                status = "✅ Positive rewards!"
            elif current_avg > -50:
                status = "📈 Improving..."
            else:
                status = "⏳ Learning..."
            
            stats_text = f"""TRAINING STATISTICS

Episodes Completed: {len(ep_rewards)}
Total Steps: {timesteps:,}

Current Avg (50): {current_avg:.2f}
Recent Avg (20): {recent_avg:.2f}
Best Avg Ever: {best_reward:.2f}

Status: {status}
"""
            
            self.axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, 
                                verticalalignment='center', family='monospace',
                                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
            
            # Update the figure title with progress
            progress_pct = (timesteps / 300000) * 100 if timesteps else 0
            self.fig.suptitle(f'PPO Training Progress - {progress_pct:.1f}% Complete', 
                            fontsize=16, fontweight='bold')
            
            # Adjust layout and redraw
            plt.tight_layout()
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            
        except Exception as e:
            # Silently ignore plotting errors to not disrupt training
            pass
    
    def close(self):
        """Close the plot window."""
        if not self.is_closed and self.fig is not None:
            try:
                plt.close(self.fig)
            except:
                pass
            self.is_closed = True
    
    def save(self, filepath):
        """Save the current plot to file."""
        if self.fig is None:
            return
            
        try:
            self.fig.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {filepath}")
        except Exception as e:
            print(f"Error saving plot: {e}")


class PlotterThread:
    """Wrapper to run plotter - handles initialization safely."""
    
    def __init__(self, max_episodes=1000):
        self.plotter = None
        self.max_episodes = max_episodes
        self.started = False
        self._init_error = None
    
    def start(self):
        """Start the plotter in main thread (matplotlib requirement)."""
        if not self.started:
            try:
                print("Initializing live plotter...")
                self.plotter = LivePlotter(max_episodes=self.max_episodes)
                self.started = True
                print("Live plotter ready - window should appear shortly")
            except Exception as e:
                self._init_error = str(e)
                print(f"Warning: Could not initialize plotter: {e}")
                self.plotter = None
    
    def update(self, episode_reward, avg_reward, num_timesteps, best_avg_reward=None):
        """Update plot with new data."""
        if self.plotter and self.started:
            return self.plotter.update(episode_reward, avg_reward, num_timesteps, best_avg_reward)
        return True
    
    def close(self):
        """Close the plotter."""
        if self.plotter:
            self.plotter.close()
    
    def save(self, filepath):
        """Save current plot."""
        if self.plotter:
            self.plotter.save(filepath)

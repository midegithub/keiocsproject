"""
Live plotting module for training visualization.
Shows: reward, distance, survival time graphs + stats

"""

import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for better Windows compatibility
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import threading


class LivePlotter:
    """Non-blocking live plotter for training metrics"""
    
    def __init__(self, max_episodes=1000):
        """
        Initialize the live plotter
        
        Args:
            max_episodes: Maximum number of episodes to display
        """
        self.max_episodes = max_episodes
        
        # Data storage
        self.episode_rewards = deque(maxlen=max_episodes)
        self.episode_distances = deque(maxlen=max_episodes)
        self.episode_times = deque(maxlen=max_episodes)
        self.episodes = deque(maxlen=max_episodes)
        
        # Best stats (independent)
        self.best_reward = float('-inf')
        self.best_distance = 0.0
        self.best_time = 0.0
        
        # Stats of best distance robot
        self.best_dist_robot_reward = 0.0
        self.best_dist_robot_time = 0.0
        self.best_dist_robot_distance = 0.0
        
        self.num_timesteps = 0
        
        # Thread-safe lock
        self.lock = threading.Lock()
        self.has_data = False
        
        # Create figure with 4 subplots (2x2)
        try:
            self.fig, self.axes = plt.subplots(2, 2, figsize=(14, 9))
            self.fig.suptitle('Training Progress - Goal: 100m straight walk!', fontsize=14, fontweight='bold')
            
            self._show_loading_screen()
            
            plt.ion()
            plt.tight_layout()
            self.fig.canvas.draw()
            plt.show(block=False)
            self.fig.canvas.flush_events()
            
        except Exception as e:
            print(f"Warning: Could not create plot window: {e}")
            self.fig = None
            self.axes = None
        
        self.is_closed = False
        
        if self.fig is not None:
            self.fig.canvas.mpl_connect('close_event', self._on_close)
    
    def _show_loading_screen(self):
        """Display loading message"""
        if self.axes is None:
            return
            
        for i, ax in enumerate(self.axes.flat):
            ax.clear()
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            
            messages = [
                'REWARD GRAPH\n\nWaiting for data...',
                'DISTANCE GRAPH\n\nWaiting for data...',
                'SURVIVAL TIME GRAPH\n\nWaiting for data...',
                'STATISTICS\n\nWaiting for data...'
            ]
            colors = ['lightblue', 'lightgreen', 'lightyellow', 'wheat']
            
            ax.text(0.5, 0.5, messages[i], ha='center', va='center', fontsize=12,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor=colors[i], alpha=0.8))
    
    def _on_close(self, event):
        """Handle plot window close"""
        self.is_closed = True
    
    def update(self, episode_reward, distance, survival_time, num_timesteps):
        """
        Update the plot with new episode data
        
        Args:
            episode_reward: Total reward for the episode
            distance: Distance traveled (meters)
            survival_time: How long robot survived (seconds)
            num_timesteps: Total training timesteps so far
        """
        if self.is_closed or self.fig is None:
            return False
        
        with self.lock:
            self.episode_rewards.append(episode_reward)
            self.episode_distances.append(distance)
            self.episode_times.append(survival_time)
            self.episodes.append(len(self.episode_rewards))
            self.num_timesteps = num_timesteps
            self.has_data = True
            
            # Update best stats (independent)
            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
            if distance > self.best_distance:
                self.best_distance = distance
                # Also save this robot's other stats
                self.best_dist_robot_reward = episode_reward
                self.best_dist_robot_time = survival_time
                self.best_dist_robot_distance = distance
            if survival_time > self.best_time:
                self.best_time = survival_time
        
        try:
            self._redraw()
            return True
        except Exception as e:
            return False
    
    def _redraw(self):
        """Redraw all plots"""
        if self.fig is None or self.axes is None or not self.has_data:
            return
        
        with self.lock:
            episodes = list(self.episodes)
            rewards = list(self.episode_rewards)
            distances = list(self.episode_distances)
            times = list(self.episode_times)
        
        if len(episodes) < 2:
            return
        
        try:
            # Clear all axes
            for ax in self.axes.flat:
                ax.clear()
            
            # ===== PLOT 1: REWARD GRAPH =====
            ax1 = self.axes[0, 0]
            ax1.plot(episodes, rewards, 'b-', alpha=0.4, linewidth=0.8, label='Episode')
            # Moving average
            if len(rewards) >= 20:
                avg = np.convolve(rewards, np.ones(20)/20, mode='valid')
                ax1.plot(episodes[19:], avg, 'r-', linewidth=2, label='Avg(20)')
            ax1.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Reward')
            ax1.set_title('Episode Rewards', fontweight='bold')
            ax1.legend(loc='upper left', fontsize=8)
            ax1.grid(True, alpha=0.3)
            
            # ===== PLOT 2: DISTANCE GRAPH (auto-scaled) =====
            ax2 = self.axes[0, 1]
            ax2.plot(episodes, distances, 'g-', alpha=0.4, linewidth=0.8, label='Episode')
            # Moving average
            if len(distances) >= 20:
                avg = np.convolve(distances, np.ones(20)/20, mode='valid')
                ax2.plot(episodes[19:], avg, 'darkgreen', linewidth=2, label='Avg(20)')
            # Best distance line
            ax2.axhline(y=self.best_distance, color='red', linestyle='--', linewidth=1.5, 
                       label=f'Best: {self.best_distance:.2f}m')
            # Auto-scale Y axis to show actual progress clearly
            y_max = max(self.best_distance * 1.5, 10.0)
            ax2.set_ylim(0, y_max)
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Distance (m)')
            ax2.set_title(f'Distance Traveled (Goal: 100m)', fontweight='bold')
            ax2.legend(loc='upper left', fontsize=8)
            ax2.grid(True, alpha=0.3)
            
            # ===== PLOT 3: SURVIVAL TIME GRAPH =====
            ax3 = self.axes[1, 0]
            ax3.plot(episodes, times, 'purple', alpha=0.4, linewidth=0.8, label='Episode')
            # Moving average
            if len(times) >= 20:
                avg = np.convolve(times, np.ones(20)/20, mode='valid')
                ax3.plot(episodes[19:], avg, 'darkviolet', linewidth=2, label='Avg(20)')
            ax3.axhline(y=self.best_time, color='red', linestyle='--', linewidth=1.5,
                       label=f'Best: {self.best_time:.1f}s')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Time (seconds)')
            ax3.set_title('Survival Time', fontweight='bold')
            ax3.legend(loc='upper left', fontsize=8)
            ax3.grid(True, alpha=0.3)
            
            # ===== PLOT 4: STATISTICS =====
            ax4 = self.axes[1, 1]
            ax4.axis('off')
            
            # Recent averages
            recent_reward = np.mean(rewards[-50:]) if len(rewards) >= 50 else np.mean(rewards)
            recent_dist = np.mean(distances[-50:]) if len(distances) >= 50 else np.mean(distances)
            recent_time = np.mean(times[-50:]) if len(times) >= 50 else np.mean(times)
            
            stats_text = f"""
╔══════════════════════════════════════════╗
║          TRAINING STATISTICS             ║
╠══════════════════════════════════════════╣
║  Episodes: {len(episodes):>6}                       ║
║  Timesteps: {self.num_timesteps:>10,}                ║
╠══════════════════════════════════════════╣
║         BEST STATS (Independent)         ║
╠──────────────────────────────────────────╣
║  Best Reward:    {self.best_reward:>10.2f}             ║
║  Best Distance:  {self.best_distance:>10.2f} m           ║
║  Best Survival:  {self.best_time:>10.1f} s           ║
╠══════════════════════════════════════════╣
║     BEST DISTANCE ROBOT's FULL STATS     ║
╠──────────────────────────────────────────╣
║  Distance:  {self.best_dist_robot_distance:>8.2f} m                 ║
║  Reward:    {self.best_dist_robot_reward:>8.2f}                   ║
║  Survived:  {self.best_dist_robot_time:>8.1f} s                 ║
╠══════════════════════════════════════════╣
║           RECENT AVERAGES (50ep)         ║
╠──────────────────────────────────────────╣
║  Avg Reward:   {recent_reward:>8.2f}                 ║
║  Avg Distance: {recent_dist:>8.2f} m               ║
║  Avg Survival: {recent_time:>8.1f} s               ║
╚══════════════════════════════════════════╝
"""
            
            ax4.text(0.05, 0.95, stats_text, fontsize=9, family='monospace',
                    verticalalignment='top', transform=ax4.transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
            
            # Update title with progress
            progress = (self.num_timesteps / 5_000_000) * 100
            self.fig.suptitle(f'Training Progress - {progress:.1f}% | Best Distance: {self.best_distance:.2f}m / 100m Goal', 
                            fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            
        except Exception as e:
            pass  # Don't crash training if plotting fails
    
    def close(self):
        """Close the plot window"""
        if not self.is_closed and self.fig is not None:
            try:
                plt.close(self.fig)
            except:
                pass
            self.is_closed = True
    
    def save(self, filepath):
        """Save the current plot to file"""
        if self.fig is None:
            return
        try:
            self.fig.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {filepath}")
        except Exception as e:
            print(f"Error saving plot: {e}")


class PlotterThread:
    """Wrapper to run plotter safely"""
    
    def __init__(self, max_episodes=1000):
        self.plotter = None
        self.max_episodes = max_episodes
        self.started = False
    
    def start(self):
        """Start the plotter"""
        if not self.started:
            try:
                print("Initializing live plotter...")
                self.plotter = LivePlotter(max_episodes=self.max_episodes)
                self.started = True
                print("Live plotter ready!")
            except Exception as e:
                print(f"Warning: Could not initialize plotter: {e}")
                self.plotter = None
    
    def update(self, episode_reward, distance, survival_time, num_timesteps):
        """Update plot with new data"""
        if self.plotter and self.started:
            return self.plotter.update(episode_reward, distance, survival_time, num_timesteps)
        return True
    
    def close(self):
        """Close the plotter"""
        if self.plotter:
            self.plotter.close()
    
    def save(self, filepath):
        """Save current plot"""
        if self.plotter:
            self.plotter.save(filepath)

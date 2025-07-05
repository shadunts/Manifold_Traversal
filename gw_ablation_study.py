import numpy as np
import os
import pickle
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

from models.manifold_traversal import ManifoldTraversal

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"]
})


class GWAblationStudy:
    """
    Conducts ablation study on GW data using manifold traversal networks.
    """

    def __init__(self, data_dir='./data', save_dir='results'):
        """
        Initialize ablation study.

        Args:
            data_dir: Directory containing GW training/test data
            save_dir: Directory to save results
        """
        self.data_dir = data_dir
        self.save_dir = save_dir

        os.makedirs(self.save_dir, exist_ok=True)

        self.train_waves_filename = 'datawaves_100000_spinsFalse_nonuniform.npy'
        self.test_waves_filename = 'datawaves_20000_spinsFalse_nonuniform.npy'

        self.sigma = 0.01
        self.d = 2

        self.D = None
        self.X_train = None
        self.X_natural_train = None
        self.X_test = None
        self.X_natural_test = None
        self.N_train = None
        self.N_test = None

        self.networks = []
        self.network_names = []
        self.network_results = []
        self.network_stats = []

    def load_data(self):
        """Load training and test data."""
        print("Loading GW training data...")
        train_file = os.path.join(self.data_dir, self.train_waves_filename)
        train_waves = np.load(train_file)

        print("Loading GW test data...")
        test_file = os.path.join(self.data_dir, self.test_waves_filename)
        test_waves = np.load(test_file)

        self.X_natural_train = train_waves.T
        self.X_natural_test = test_waves.T

        self.N_train = self.X_natural_train.shape[1]
        self.N_test = self.X_natural_test.shape[1]
        self.D = self.X_natural_train.shape[0]

        self.X_train = self.X_natural_train + self.sigma * np.random.randn(self.D, self.N_train)
        self.X_test = self.X_natural_test + self.sigma * np.random.randn(self.D, self.N_test)

        print(f"Training data shape: {self.X_train.shape}")
        print(f"Test data shape: {self.X_test.shape}")
        print(f"Ambient dimension (D): {self.D}")
        print(f"Intrinsic dimension (d): {self.d}")
        print(f"Noise level (sigma): {self.sigma}")

    def get_hyperparameter_configs(self):
        """
        Get the same 12 hyperparameter configurations as the original ablation study.

        Returns:
            List of dictionaries containing hyperparameters for each network
        """
        configs = []

        sigma_sq_D = self.sigma ** 2 * self.D
        sigma_sq_d = self.sigma ** 2 * self.d

        base_configs = [
            # Network 1
            {
                'R_is_const': False,
                'R_denoising': np.sqrt(2.06 * sigma_sq_D),
                'R_1st_order_nbhd': np.sqrt(2.39 * sigma_sq_D),
                'd_parallel': np.sqrt(20 * sigma_sq_d),
                'prod_coeff': 1.2,
                'exp_coeff': 1 / 2
            },
            # Network 2
            {
                'R_is_const': True,
                'R_denoising': np.sqrt(2.06 * sigma_sq_D),
                'R_1st_order_nbhd': np.sqrt(2.39 * sigma_sq_D),
                'd_parallel': np.sqrt(20 * sigma_sq_d),
                'prod_coeff': 1.2,
                'exp_coeff': 1 / 2
            },
            # Network 3
            {
                'R_is_const': False,
                'R_denoising': np.sqrt(2.06 * sigma_sq_D),
                'R_1st_order_nbhd': np.sqrt(2.39 * sigma_sq_D),
                'd_parallel': np.sqrt(8 * sigma_sq_d),
                'prod_coeff': 1.2,
                'exp_coeff': 1 / 2
            },
            # Network 4
            {
                'R_is_const': True,
                'R_denoising': np.sqrt(2.75 * sigma_sq_D),
                'R_1st_order_nbhd': np.sqrt(2.75 * sigma_sq_D),
                'd_parallel': np.sqrt(20 * sigma_sq_d),
                'prod_coeff': 1.2,
                'exp_coeff': 1 / 2
            },
            # Network 5
            {
                'R_is_const': False,
                'R_denoising': np.sqrt(2.06 * sigma_sq_D),
                'R_1st_order_nbhd': np.sqrt(2.39 * sigma_sq_D),
                'd_parallel': np.sqrt(20 * sigma_sq_d),
                'prod_coeff': 1.3,
                'exp_coeff': 1 / 3
            },
            # Network 6
            {
                'R_is_const': False,
                'R_denoising': np.sqrt(2.06 * sigma_sq_D),
                'R_1st_order_nbhd': np.sqrt(2.39 * sigma_sq_D),
                'd_parallel': np.sqrt(4 * sigma_sq_d),
                'prod_coeff': 1.15,
                'exp_coeff': 1 / 2
            },
            # Network 7
            {
                'R_is_const': True,
                'R_denoising': np.sqrt(2.39 * sigma_sq_D),
                'R_1st_order_nbhd': np.sqrt(2.75 * sigma_sq_D),
                'd_parallel': np.sqrt(20 * sigma_sq_d),
                'prod_coeff': 1.2,
                'exp_coeff': 1 / 2
            },
            # Network 8
            {
                'R_is_const': False,
                'R_denoising': np.sqrt(2.06 * sigma_sq_D),
                'R_1st_order_nbhd': np.sqrt(2.39 * sigma_sq_D),
                'd_parallel': np.sqrt(30 * sigma_sq_d),
                'prod_coeff': 1.5,
                'exp_coeff': 1 / 2
            },
            # Network 9
            {
                'R_is_const': True,
                'R_denoising': np.sqrt(2 * sigma_sq_D),
                'R_1st_order_nbhd': np.sqrt(2.39 * sigma_sq_D),
                'd_parallel': np.sqrt(20 * sigma_sq_d),
                'prod_coeff': 1.2,
                'exp_coeff': 1 / 2
            },
            # Network 10
            {
                'R_is_const': True,
                'R_denoising': np.sqrt(2.19 * sigma_sq_D),
                'R_1st_order_nbhd': np.sqrt(2.39 * sigma_sq_D),
                'd_parallel': np.sqrt(20 * sigma_sq_d),
                'prod_coeff': 1.2,
                'exp_coeff': 1 / 2
            },
            # Network 11
            {
                'R_is_const': True,
                'R_denoising': np.sqrt(3.13 * sigma_sq_D),
                'R_1st_order_nbhd': np.sqrt(3.53 * sigma_sq_D),
                'd_parallel': np.sqrt(20 * sigma_sq_d),
                'prod_coeff': 1.2,
                'exp_coeff': 1 / 2
            },
            # Network 12
            {
                'R_is_const': True,
                'R_denoising': np.sqrt(1.94 * sigma_sq_D),
                'R_1st_order_nbhd': np.sqrt(2.39 * sigma_sq_D),
                'd_parallel': np.sqrt(20 * sigma_sq_d),
                'prod_coeff': 1.2,
                'exp_coeff': 1 / 2
            }
        ]

        return base_configs

    def train_networks(self, batch_size=4000):
        """
        Train all networks with different hyperparameter configurations.

        Args:
            batch_size: Batch size for training progress reporting
        """
        if self.X_train is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        configs = self.get_hyperparameter_configs()

        print(f"Training {len(configs)} networks with different hyperparameters...")
        print("=" * 80)

        for i, config in enumerate(configs):
            network_name = f"NETWORK_{i + 1}"
            print(f"\nTraining {network_name}")
            print(f"Config: {config}")
            print("-" * 60)

            mt = ManifoldTraversal(
                intrinsic_dim=self.d,
                ambient_dim=self.D,
                sigma=self.sigma,
                **config
            )

            start_time = time.time()
            results = mt.fit(self.X_train, self.X_natural_train,
                             batch_size=batch_size, verbose=True)
            training_time = time.time() - start_time

            self.networks.append(mt)
            self.network_names.append(network_name)
            self.network_results.append(results)

            stats = mt.network.get_network_stats()
            stats['training_time'] = training_time
            stats['config'] = config
            self.network_stats.append(stats)

            print(f"Training completed in {training_time:.2f}s")
            print(f"Network stats: {stats}")
            print("~" * 60)

        print(f"\nAll {len(configs)} networks trained successfully!")

    def analyze_networks(self, num_test_samples=None):
        """
        Analyze performance of all trained networks.

        Args:
            num_test_samples: Number of test samples to use (None for all)
        """
        if not self.networks:
            raise ValueError("No networks trained. Call train_networks() first.")

        if num_test_samples is None:
            num_test_samples = self.N_test

        print(f"\nAnalyzing performance on {num_test_samples} test samples...")
        print("=" * 80)

        analysis_results = []

        for i, (mt, name) in enumerate(zip(self.networks, self.network_names)):
            print(f"Analyzing {name}...")

            results = mt.analyze_performance(
                self.X_test, self.X_natural_test,
                num_samples=num_test_samples
            )

            analysis_results.append(results)

            print(f"  Exhaustive Search: Error={results['exhaustive']['avg_distance']:.6f}, "
                  f"Complexity={results['exhaustive']['avg_mults']:.1f}")
            print(f"  Mixed Order (MT): Error={results['mixed_order']['avg_distance']:.6f}, "
                  f"Complexity={results['mixed_order']['avg_mults']:.1f}")
            print(f"  First Order Only: Error={results['first_order_only']['avg_distance']:.6f}, "
                  f"Complexity={results['first_order_only']['avg_mults']:.1f}")
            print(f"  Zero Order Only: Error={results['zero_order_only']['avg_distance']:.6f}, "
                  f"Complexity={results['zero_order_only']['avg_mults']:.1f}")
            print()

        self.analysis_results = analysis_results
        print("Analysis complete!")

    def plot_training_curve(self, network_idx=0, save_path=None):
        """
        Plot training error curve for a specific network.

        Args:
            network_idx: Index of network to plot (default: first network)
            save_path: Path to save plot (optional)
        """
        if network_idx >= len(self.network_results):
            raise ValueError(f"Network index {network_idx} out of range")

        results = self.network_results[network_idx]

        sigma_sq_d = [self.sigma ** 2 * self.d] * len(results.mean_MT_error)
        sigma_sq_D = [self.sigma ** 2 * self.D] * len(results.mean_MT_error)

        plt.figure(figsize=(10, 6))
        plt.plot(results.mean_MT_error, color='orange', label='MT training')
        plt.plot(sigma_sq_d, color='green', label=r'$\sigma^2 d$')
        plt.plot(sigma_sq_D, color='blue', label=r'$\sigma^2 D$')
        plt.legend()
        plt.xlabel('Number of Samples')
        plt.ylabel('Mean Square Error (Train)')
        plt.title(f'Training Error Curve - {self.network_names[network_idx]}')
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

        if save_path:
            plt.savefig(save_path, format='pdf', bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_complexity_accuracy_tradeoff(self, save_path=None):
        """
        Plot computational complexity vs accuracy tradeoff (MT vs NN).

        Args:
            save_path: Path to save plot (optional)
        """
        if not hasattr(self, 'analysis_results'):
            raise ValueError("Analysis not performed. Call analyze_networks() first.")

        configs = self.get_hyperparameter_configs()
        mt_accuracies = [r['mixed_order']['avg_distance'] for r in self.analysis_results]
        mt_complexities = [r['mixed_order']['avg_mults'] for r in self.analysis_results]
        nn_accuracies = [r['exhaustive']['avg_distance'] for r in self.analysis_results]
        nn_complexities = [r['exhaustive']['avg_mults'] for r in self.analysis_results]

        plt.figure(figsize=(10, 6))

        for i in range(len(self.networks)):
            marker_mt = 'o' if configs[i]['R_is_const'] else '^'
            plt.scatter(mt_accuracies[i], np.log10(mt_complexities[i]),
                        marker=marker_mt, color='blue', s=60)
            plt.scatter(nn_accuracies[i], np.log10(nn_complexities[i]),
                        marker='s', color='red', s=60)

        plt.title('Computational Complexity vs. Accuracy')
        plt.ylabel("log of (\#Multiplications per sample)")
        plt.xlabel("Mean Squared Error")

        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Manifold Traversal (R constant)',
                   markerfacecolor='blue', markersize=8),
            Line2D([0], [0], marker='^', color='w', label='Manifold Traversal (R decreasing)',
                   markerfacecolor='blue', markersize=8),
            Line2D([0], [0], marker='s', color='w', label='Nearest Neighbor',
                   markerfacecolor='red', markersize=8),
        ]
        plt.legend(handles=legend_elements)
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

        if save_path:
            plt.savefig(save_path, format='pdf', bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_ablation_study(self, save_path=None):
        """
        Plot ablation study comparing mixed-order, first-order, and zero-order methods.

        Args:
            save_path: Path to save plot (optional)
        """
        if not hasattr(self, 'analysis_results'):
            raise ValueError("Analysis not performed. Call analyze_networks() first.")

        configs = self.get_hyperparameter_configs()
        mt_accuracies = [r['mixed_order']['avg_distance'] for r in self.analysis_results]
        mt_complexities = [r['mixed_order']['avg_mults'] for r in self.analysis_results]
        fom_accuracies = [r['first_order_only']['avg_distance'] for r in self.analysis_results]
        fom_complexities = [r['first_order_only']['avg_mults'] for r in self.analysis_results]
        zom_accuracies = [r['zero_order_only']['avg_distance'] for r in self.analysis_results]
        zom_complexities = [r['zero_order_only']['avg_mults'] for r in self.analysis_results]

        plt.figure(figsize=(10, 6))

        for i in range(len(self.networks)):
            marker_mt = 'o' if configs[i]['R_is_const'] else '^'
            plt.scatter(mt_accuracies[i], np.log10(mt_complexities[i]),
                        marker=marker_mt, color='blue', s=60)
            plt.scatter(fom_accuracies[i], np.log10(fom_complexities[i]),
                        marker='*', color='orange', s=60)
            plt.scatter(zom_accuracies[i], np.log10(zom_complexities[i]),
                        marker='x', color='purple', s=60)

        plt.title('Ablation Study of Mixed-Order Method')
        plt.ylabel("log of (\#Multiplications per sample)")
        plt.xlabel("Mean Squared Error")

        legend_elements = [
            Line2D([0], [0], marker='o', color='w',
                   label='Mixed-Order Optimization (R constant)',
                   markerfacecolor='blue', markersize=8),
            Line2D([0], [0], marker='^', color='w',
                   label='Mixed-Order Optimization (R decreasing)',
                   markerfacecolor='blue', markersize=8),
            Line2D([0], [0], marker='*', color='w',
                   label='First-Order Optimization',
                   markerfacecolor='orange', markersize=8),
            Line2D([0], [0], marker='x', color='w',
                   label='Zero-Order Optimization',
                   markeredgecolor='purple', markersize=8, linestyle='None'),
        ]
        plt.legend(handles=legend_elements)
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

        if save_path:
            plt.savefig(save_path, format='pdf', bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_ablation_study_zoom(self, x_range=(0, 0.06), y_range=(4, 7), save_path=None):
        """
        Plot zoomed-in version of ablation study.

        Args:
            x_range: X-axis range for zoom
            y_range: Y-axis range for zoom
            save_path: Path to save plot (optional)
        """
        if not hasattr(self, 'analysis_results'):
            raise ValueError("Analysis not performed. Call analyze_networks() first.")

        configs = self.get_hyperparameter_configs()
        mt_accuracies = [r['mixed_order']['avg_distance'] for r in self.analysis_results]
        mt_complexities = [r['mixed_order']['avg_mults'] for r in self.analysis_results]
        zom_accuracies = [r['zero_order_only']['avg_distance'] for r in self.analysis_results]
        zom_complexities = [r['zero_order_only']['avg_mults'] for r in self.analysis_results]

        plt.figure(figsize=(10, 6))

        for i in range(len(self.networks)):
            marker_mt = 'o' if configs[i]['R_is_const'] else '^'
            plt.scatter(mt_accuracies[i], np.log10(mt_complexities[i]),
                        marker=marker_mt, color='blue', s=60)
            plt.scatter(zom_accuracies[i], np.log10(zom_complexities[i]),
                        marker='x', color='purple', s=60)

        plt.title('Ablation Study of Mixed-Order Method (zoom in)')
        plt.ylabel("log of (\#Multiplications per sample)")
        plt.xlabel("Mean Squared Error")

        legend_elements = [
            Line2D([0], [0], marker='o', color='w',
                   label='Mixed-Order Optimization (R constant)',
                   markerfacecolor='blue', markersize=8),
            Line2D([0], [0], marker='^', color='w',
                   label='Mixed-Order Optimization (R decreasing)',
                   markerfacecolor='blue', markersize=8),
            Line2D([0], [0], marker='x', color='w',
                   label='Zero-Order Optimization',
                   markeredgecolor='purple', markersize=8, linestyle=None),
        ]
        plt.legend(handles=legend_elements)
        plt.xlim(x_range)
        plt.ylim(y_range)
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

        if save_path:
            plt.savefig(save_path, format='pdf', bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def save_results(self, filename='gw_ablation_results.pkl'):
        """
        Save all results to pickle file.

        Args:
            filename: Name of file to save results
        """
        # extract only picklable data from networks
        network_configs = []
        for i, mt in enumerate(self.networks):
            config = {
                'intrinsic_dim': mt.intrinsic_dim,
                'ambient_dim': mt.ambient_dim,
                'sigma': mt.sigma,
                'R_is_const': mt.R_is_const,
                'R_denoising': mt.R_denoising,
                'R_1st_order_nbhd': mt.R_1st_order_nbhd,
                'd_parallel': mt.d_parallel,
                'prod_coeff': mt.prod_coeff,
                'exp_coeff': mt.exp_coeff
            }
            network_configs.append(config)

        # extract data from network_results
        training_summaries = []
        for result in self.network_results:
            if hasattr(result, 'mean_MT_error') and result.mean_MT_error is not None:
                summary = {
                    'final_error': result.mean_MT_error[-1] if len(result.mean_MT_error) > 0 else None,
                    'error_history': result.mean_MT_error,
                    'num_epochs': len(result.mean_MT_error) if result.mean_MT_error else 0
                }
            else:
                summary = {
                    'final_error': None,
                    'error_history': [],
                    'num_epochs': 0
                }
            training_summaries.append(summary)

        # extract only numeric data from network_stats to avoid circular references
        safe_network_stats = []
        for stats in self.network_stats:
            safe_stats = {}
            for key, value in stats.items():
                if isinstance(value, (int, float, bool, type(None))):
                    safe_stats[key] = value
                elif isinstance(value, np.ndarray):
                    safe_stats[key] = value.tolist()
                elif isinstance(value, (list, tuple)) and all(isinstance(x, (int, float)) for x in value):
                    safe_stats[key] = list(value)
                else:
                    # Skip complex objects that might have circular references
                    safe_stats[key] = str(type(value).__name__)
            safe_network_stats.append(safe_stats)

        # extract only numeric data from analysis_results
        safe_analysis_results = None
        if hasattr(self, 'analysis_results') and self.analysis_results:
            safe_analysis_results = []
            for result in self.analysis_results:
                safe_result = {}
                for method, data in result.items():
                    if isinstance(data, dict):
                        safe_data = {}
                        for key, value in data.items():
                            if isinstance(value, (int, float, bool, type(None))):
                                safe_data[key] = value
                            elif isinstance(value, np.ndarray):
                                safe_data[key] = value.tolist()
                            elif isinstance(value, (list, tuple)) and all(isinstance(x, (int, float)) for x in value):
                                safe_data[key] = list(value)
                        safe_result[method] = safe_data
                safe_analysis_results.append(safe_result)

        results_dict = {
            'network_names': self.network_names,
            'network_configs': network_configs,
            'training_summaries': training_summaries,
            'network_stats': safe_network_stats,
            'analysis_results': safe_analysis_results,
            'hyperparameter_configs': self.get_hyperparameter_configs(),
            'sigma': self.sigma,
            'D': self.D,
            'd': self.d,
            'N_train': self.N_train,
            'N_test': self.N_test
        }

        save_path = os.path.join(self.save_dir, filename)
        try:
            with open(save_path, 'wb') as f:
                pickle.dump(results_dict, f)
            print(f"Results saved to {save_path}")
        except Exception as e:
            print(f"Error saving results: {e}")
            # fallback: save as JSON
            json_filename = filename.replace('.pkl', '.json')
            json_path = os.path.join(self.save_dir, json_filename)
            try:
                import json

                def make_json_serializable(obj):
                    """Recursively convert objects to JSON-serializable format."""
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
                        return obj.item()
                    elif isinstance(obj, dict):
                        return {k: make_json_serializable(v) for k, v in obj.items()}
                    elif isinstance(obj, (list, tuple)):
                        return [make_json_serializable(item) for item in obj]
                    elif isinstance(obj, (int, float, str, bool)) or obj is None:
                        return obj
                    else:
                        return str(obj)

                json_dict = make_json_serializable(results_dict)

                with open(json_path, 'w') as f:
                    json.dump(json_dict, f, indent=2)
                print(f"Results also saved as JSON to {json_path}")
            except Exception as json_error:
                print(f"Error saving JSON backup: {json_error}")

    def load_results(self, filename='gw_ablation_results.pkl'):
        """
        Load results from pickle file.

        Args:
            filename: Name of file to load results from
        """
        load_path = os.path.join(self.save_dir, filename)

        # try to load pickle file first
        try:
            with open(load_path, 'rb') as f:
                results_dict = pickle.load(f)
        except FileNotFoundError:
            # try JSON fallback
            json_filename = filename.replace('.pkl', '.json')
            json_path = os.path.join(self.save_dir, json_filename)
            try:
                import json
                with open(json_path, 'r') as f:
                    results_dict = json.load(f)
                print(f"Loaded from JSON fallback: {json_path}")
            except FileNotFoundError:
                raise FileNotFoundError(f"Neither {load_path} nor {json_path} found")

        # handle both old and new format
        if 'networks' in results_dict:
            # Old format - try to load directly
            self.networks = results_dict['networks']
            self.network_results = results_dict['network_results']
        else:
            # New format - networks and results are not saved
            print("Note: Full network objects not available in saved results.")
            print("Only configuration and analysis data loaded.")
            self.networks = []
            self.network_results = []

        self.network_names = results_dict['network_names']
        self.network_stats = results_dict['network_stats']
        self.analysis_results = results_dict.get('analysis_results', None)
        self.sigma = results_dict['sigma']
        self.D = results_dict['D']
        self.d = results_dict['d']

        # handle optional fields that might not be in older saves
        self.N_train = results_dict.get('N_train', None)
        self.N_test = results_dict.get('N_test', None)

        print(f"Results loaded from {load_path}")

        # if we have analysis results, print a summary
        if self.analysis_results:
            print(f"Loaded analysis results for {len(self.analysis_results)} networks")


def run_full_ablation_study():
    """
    Run the complete GW ablation study.
    """
    print("=" * 80)
    print("GW MANIFOLD TRAVERSAL ABLATION STUDY")
    print("=" * 80)

    study = GWAblationStudy()

    study.load_data()

    study.train_networks(batch_size=4000)

    study.analyze_networks()

    print("\nGenerating plots...")
    study.plot_training_curve(0, save_path='results/training_error_curve.pdf')
    study.plot_complexity_accuracy_tradeoff(save_path='results/MT_NN_tradeoff.pdf')
    study.plot_ablation_study(save_path='results/ablation_study.pdf')

    try:
        study.save_results()
    except Exception as e:
        print(f"Warning: Could not save results due to: {e}")
        print("Plots have been generated successfully.")

    print("\nAblation study complete!")
    return study


if __name__ == "__main__":
    study = run_full_ablation_study()

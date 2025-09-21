import numpy as np
import matplotlib.pyplot as plt
from fallback_cosmology import Cosmology  

class Likelihood:
    """
    Likelihood class that calculates the log likelihood with the use of a covariance matrix. The likelihood marginalisation, likelihood grids and
    plotting the likelihood graphs was removed as it is no longer needed for unit 6.
    """
    def __init__(self, data_file, covariance_file):
        """
        Initializes the likelihood instance by loading the observed data and covariance matrix.
        
        Parameters
        ----------
        data_file : str
            Path to the supernova data file.
        covariance_file : str
            Path to the covariance matrix file.
        """
        data = np.loadtxt(data_file)
        self.z = data[:, 0]
        self.m_obs = data[:, 1]
        
        # Load covariance matrix and compute its inverse
        self.cov_matrix = np.loadtxt(covariance_file)
        self.inv_cov_matrix = np.linalg.inv(self.cov_matrix)

    def compute_model(self, Omega_m, Omega_lambda, H0):
        """
        Computes magnitudes using the fallback cosmology class.
        
        parameters
        ----------
        Omega_m : float
        Omega_lambda : float
        H0 : float
        
        Returns
        -------
        ndarray
            Magnitudes for the given cosmological parameters.
        """
        cosmology = Cosmology(Omega_m, Omega_lambda, H0)
        mu = cosmology.distance_modulus(self.z)
        return mu - 19.3

    def log_likelihood(self, theta):
        """
        Calculates the log-likelihood for the given cosmological parameters using the covariance matrix.
        
        Parameters
        ----------
        theta : list
            A list of cosmological parameters 
            
        Returns
        -------
        float
            The log-likelihood.
        """
        Omega_m, Omega_lambda, H0 = theta
        m_model = self.compute_model(Omega_m, Omega_lambda, H0)
        residual = self.m_obs - m_model
        
        # Compute likelihood using covariance matrix
        chi_squared = residual.T @ self.inv_cov_matrix @ residual
        return -0.5 * chi_squared

class Metropolis:
    """
    This class uses the Metropolis-Hastings algorithm for sampling from a given likelihood function.
    """
    def __init__(self, likelihood_function, initial_position, step_size, num_samples):
        """
        Initializes the Metropolis instance. 
        """
        self.likelihood_function = likelihood_function
        self.current_position = np.array(initial_position)
        self.step_size = step_size
        self.num_samples = num_samples
        self.samples = []

    def propose(self):
        """
        This function generates a new sample using a Gaussian proposal distribution
        but ensures that Omega_m remains non-negative.
        
        Returns
        -------
        ndarray
            Proposed sample.
        """
        proposal = self.current_position + np.random.normal(scale=self.step_size, size=self.current_position.shape)
        
        # Ensure Omega_m is non-negative
        proposal[0] = max(proposal[0], 0.0)
        
        return proposal


    def run(self, burn_in=20000):
        """
        Runs the Metropolis-Hastings algorithm to sample from the likelihood function. It also 
        implements a burn-in period to discard initial samples.
        
        Parameters
        ----------
        burn_in : int
            The number of initial samples to discard.
        """
        current_likelihood = self.likelihood_function(self.current_position)
        self.samples.append(self.current_position)

        # '_' os a throwaway variable, it is used when there is no need in a variable
        for _ in range(self.num_samples + burn_in): 
            proposal = self.propose()
            proposed_likelihood = self.likelihood_function(proposal)

            # Computing the acceptance ratio (likelihood ratio of new vs current sample)
            acceptance_ratio = np.exp(proposed_likelihood - current_likelihood)
            # Accept or reject the proposal
            if np.random.rand() < acceptance_ratio:
                self.current_position = proposal
                current_likelihood = proposed_likelihood

            self.samples.append(self.current_position)

        # Removes the burn_in samples
        self.samples = np.array(self.samples[burn_in:])
            
    def report_statistics(self):
        """
        Prints the final mean and standard deviation for each sampled parameter.
        """
        # This line was added for debugging purposes to check whether the samples generated are empty or not. 
        if len(self.samples) == 0:
            print("No samples available, run the sampler first.")
            return

        means = np.mean(self.samples, axis=0)
        stds = np.std(self.samples, axis=0)
        labels = ["Ω_m", "Ω_Λ", "H0"]

        print("\nFinal Parameter Estimates from Metropolis Sampling:")
        for i in range(3):
            print(f"{labels[i]}: mean = {means[i]:.5f}, std = {stds[i]:.5f}")
   
    def plot_samples(self):
        """
        Plots histograms of the sampled parameters in 1D, 2D histograms, and a 3D scatter plot of 3 cosmological constants. I decided to plot these graphs individually to make it easier
        for me to add the plots into my report, eventhough there is a more efficient way to code this. 
        """
        labels = ["Ω_m", "Ω_Λ", "H0"]

        # 1D Histograms
        for i in range(self.samples.shape[1]):
            plt.figure(figsize=(6, 4))
            plt.hist(self.samples[:, i], bins=50, alpha=0.7)
            plt.xlabel(labels[i])
            plt.ylabel("Frequency")
            plt.title(f"Histogram of {labels[i]}")
            plt.tight_layout()
            plt.show()

        # 2D Histogram: Omega_m vs Omega_lambda
        plt.figure(figsize=(6, 5))
        h1 = plt.hist2d(self.samples[:, 0], self.samples[:, 1], bins=50, cmap='viridis')
        plt.xlabel("Ω_m")
        plt.ylabel("Ω_Λ")
        plt.title("2D Histogram of Ω_m vs Ω_Λ")
        plt.colorbar(h1[3], label='Probability Density')
        plt.tight_layout()
        plt.show()

        # 2D Histogram: Omega_m vs H0
        plt.figure(figsize=(6, 5))
        h2 = plt.hist2d(self.samples[:, 0], self.samples[:, 2], bins=50, cmap='viridis')
        plt.xlabel("Ω_m")
        plt.ylabel("H0")
        plt.title("2D Histogram of Ω_m vs H0")
        plt.colorbar(h2[3], label='Probability Density')
        plt.tight_layout()
        plt.show()

        # 2D Histogram: Omega_lambda vs H0
        plt.figure(figsize=(6, 5))
        h3 = plt.hist2d(self.samples[:, 1], self.samples[:, 2], bins=50, cmap='viridis')
        plt.xlabel("Ω_Λ")
        plt.ylabel("H0")
        plt.title("2D Histogram of Ω_Λ vs H0")
        plt.colorbar(h3[3], label='Probability Density')
        plt.tight_layout()
        plt.show()

        # 2D Scatter Plot
        plt.figure(figsize=(7, 6))
        scatter = plt.scatter(self.samples[:, 0], self.samples[:, 1], c=self.samples[:, 2], cmap='viridis', alpha=0.5)
        plt.xlabel("Ω_m")
        plt.ylabel("Ω_Λ")
        plt.title("2D Scatter Plot of 3 Cosmological Parameters")
        plt.colorbar(scatter, label='H0')
        plt.tight_layout()
        plt.show()

def main():
    data_file = "pantheon_data.txt"
    covariance_file = "pantheon_covariance.txt"  
    likelihood = Likelihood(data_file, covariance_file)

    # Metropolis analysis
    initial_guess = [0.3, 0.7, 70.0]
    step_size = [0.005, 0.005, 0.1]
    num_samples = 100000

    metropolis_sampler = Metropolis(likelihood.log_likelihood, initial_guess, step_size, num_samples)
    # I decided to go with a burn_in period of 20% of my num_samples to allow convergance before collecting samples
    metropolis_sampler.run(burn_in=20000)
    metropolis_sampler.plot_samples()
    metropolis_sampler.report_statistics()

if __name__ == "__main__":
    main()
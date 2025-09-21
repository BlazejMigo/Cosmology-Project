import sys
try:
    import astropy.cosmology
except ImportError:
    print("To use this fallback cosmology class you need to install astropy.")
    print("Do that by running 'conda install astropy' in the VSCode terminal window.")
    print("If that doesn't work you can try 'pip install astropy'.")
    sys.exit(1)

class Cosmology:
    """
    Class representing a standard cosmological model.

    This slimmed down version of the class doesn't have the various
    tools you created in the previous unit.  Instead it uses the
    "astropy" library to do the distance modulus calculation, which
    is the only thing you will need for units 4 and above.
    """
    def __init__(self, omega_m, omega_lambda, H0):
        """
        Create a Cosmology instance.

        Parameters
        ----------
        omega_m : float
            The fraction of the critical density that is made up of
            matter.

        omega_lambda : float
            The fraction of the critical density that is made up of
            dark energy.

        H0 : float
            The Hubble parameter in units of km/s/Mpc.
        """
        self.omega_m = omega_m
        self.omega_lambda = omega_lambda
        self.omega_k = 1.0 - omega_m - omega_lambda
        self.H0 = H0
        
        # This attribute is an astropy cosmology object. It is a bit like the
        # object you created in the earlier units.
        self._cosmo = astropy.cosmology.LambdaCDM(Om0=omega_m, Ode0=omega_lambda, H0=H0)

    def __str__(self):
        """
        Return a string representation of the Cosmology instance.

        Parameters
        ----------
        None

        Returns
        -------
        str
            A string representation of the Cosmology instance.
        """
        return f"Cosmology(omega_m={self.omega_m}, omega_lambda={self.omega_lambda}, H0={self.H0})"
    
    def distance_modulus(self, z):
        """
        Calculate the distance modulus at redshift(s) z.

        Parameters
        ----------
        z : float or array
            The redshift(s) at which to calculate the distance modulus.
            You can either use a single value or (more useful here) an array of values

        Returns
        -------
        float or array
            The distance modulus at the given redshift(s). This is dimensionless.
        """
        return self._cosmo.distmod(z).value



def test_cosmology():
    # use the three test cosmologies given in the notes
    cosmo1 = Cosmology(0.25, 0.7, 60.)
    cosmo2 = Cosmology(0.3, 0.7, 70.)
    cosmo3 = Cosmology(0.4, 0.65, 80.)

    # The three redshifts given in the notes
    mu1 = cosmo1.distance_modulus(0.5)
    mu2 = cosmo2.distance_modulus(1.0)
    mu3 = cosmo3.distance_modulus(1.5)

    # These are the values I get on my machine with my version of astropy
    expected1 = 42.615802574441574
    expected2 = 44.10023765554372
    expected3 = 44.76602040622335

    # The smallest error bar in our supernova data
    sigma = 0.2
    
    # The fractional difference from the expected values.
    # We don't expect the values to be absolutely identical between computers
    diff1 = abs(expected1 - mu1) / sigma
    diff2 = abs(expected2 - mu2) / sigma
    diff3 = abs(expected3 - mu3) / sigma

    # we will print a friendly message explaining whether or not the above differences are fine
    okay1 = "fine" if diff1 < 0.02 else "not good enough - please report this to Joe"
    okay2 = "fine" if diff2 < 0.02 else "not good enough - please report this to Joe"
    okay3 = "fine" if diff3 < 0.02 else "not good enough - please report this to Joe"

    print(f"mu1 = {mu1} - difference from expected is {diff1:f}% of error bar, which is {okay1}")
    print(f"mu1 = {mu2} - difference from expected is {diff2:f}% of error bar, which is {okay2}")
    print(f"mu1 = {mu3} - difference from expected is {diff3:f}% of error bar, which is {okay3}")




if __name__ == '__main__':
    test_cosmology()



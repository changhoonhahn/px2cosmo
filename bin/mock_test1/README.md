# Test 1 (Issue #2)
Second test using a forward model that generates $M_{UV}, z$ with **homoskedastic noise**. This is another idealized test to validate a simplified version of the idea. 

The goal is to infer the posterior on $\phi$
$$p(\phi | \{X_i\}, S) \propto p(\phi) p(N|\phi) \prod_i p(X_i | \phi, S)$$
using the individual likelihoods $p(X_i | \phi, S)$, which we can estimate with an NDE and the poisson-like likelihood, $p(N|\phi)$ also estimated using NDE . 


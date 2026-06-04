# Test 0 (Issue #1)
First test using a forward model that generates $M_{UV}, z$ with **no noise**. This is an idealized test to validate a simplified version of the idea. 

The goal is to infer the posterior on $\phi$
$$p(\phi | \{X_i\}, S) \propto p(\phi) \prod_i p(X_i | \phi, S)$$
using the individual likelihoods $p(X_i | \phi, S)$, which we can estimate with an NDE


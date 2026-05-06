#include "problems/lattice_reduction.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>

#include "cond_unknown.h"
#include "fplll_impl.h"
#include "irregular.h"

#include "lagrange.h"
#include "schoenhage.h"

#include "heuristic_1.h"
#include "heuristic_2.h"
#include "heuristic_3.h"
#include "threaded_3.h"

#include "proved_1.h"
#include "proved_2.h"
#include "proved_3.h"
#include "latred_relative_sr.h"

namespace flatter {

unsigned int LatticeReduction::policy = 2;

LatticeReduction::LatticeReduction() {
    _is_configured = false;
}

LatticeReduction::LatticeReduction(const LatticeReductionParams& p,
                                    const ComputationContext& cc) :
    Base(p, cc) 
{
    _is_configured = false;
    configure(p, cc);
}

LatticeReduction::LatticeReduction(const Matrix& M, const Matrix& U,
                                    const ComputationContext& cc) :
    LatticeReduction(LatticeReductionParams(M, U), cc)
{}

LatticeReduction::~LatticeReduction() {
    if (_is_configured) {
        unconfigure();
    }
}

void LatticeReduction::unconfigure() {
    assert(_is_configured);

    delete latred;

    _is_configured = false;
}

void LatticeReduction::configure(const LatticeReductionParams& p,
                                 const ComputationContext& cc) {
    if (_is_configured) {
        unconfigure();
    }
    Base::configure(p, cc);
    latred = nullptr;


    // Define environment variable 'FLATTER_NOFPLLL' to avoid FPLLL usage.
    bool use_fplll = std::getenv("FLATTER_NOFPLLL") == nullptr;

    // There are only 4 phases.
    assert(0 <= p.phase && p.phase <= 3);

    if (n <= 2) {
        // Base case: dimension <= 2.
        if (p.B2.nrows() != 0) {
            latred = new LatticeReductionImpl::LatRedRelSR(p, cc);
        } else if (prec < 1400 && n == 2 && m == 2) {
            latred = new LatticeReductionImpl::Lagrange(p, cc);
        } else {
            latred = new LatticeReductionImpl::Schoenhage(p, cc);
        }
    } else if (p.proved) {
        // Proven LLL-reduction
        if (p.B2.nrows() != 0) {
            latred = new LatticeReductionImpl::LatRedRelSR(p, cc);
        } else if (p.is_upper_triangular() && p.lvalid > 0 && p.lvalid == p.rvalid) {
            latred = new LatticeReductionImpl::Proved3(p, cc);
        } else if (p.is_upper_triangular()) {
            latred = new LatticeReductionImpl::Proved2(p, cc);
        } else {
            latred = new LatticeReductionImpl::Proved1(p, cc);
        }
    } else if (p.phase == 0) {
        // Phase 0: irregular lattice reduction
        latred = new LatticeReductionImpl::Irregular(p, cc);
    } else if (p.phase == 1) {
        // Phase 1: heuristic lattice reduction
        if (p.log_cond == 0) {
            // Condition number is unknown
            latred = new LatticeReductionImpl::CondUnknown(p, cc);
        } else {
            // Condition number is known, basis is guaranteed to be nonsingular
            latred = new LatticeReductionImpl::Heuristic1(p, cc);
        }
    } else if (n <= 32 && prec <= 128 && use_fplll) {
        if (p.B2.nrows() != 0) {
            latred = new LatticeReductionImpl::LatRedRelSR(p, cc);
        } else {
            // Reduce small lattices using FPLLL
            latred = new LatticeReductionImpl::FPLLL(p, cc);
        }
    } else if (p.phase == 2) {
        // Phase 2: heuristic lattice reduction
        latred = new LatticeReductionImpl::Heuristic2(p, cc);
    } else if (p.phase == 3) {
        if (p.B2.nrows() != 0) {
            latred = new LatticeReductionImpl::LatRedRelSR(p, cc);
        } else if (cc.is_threaded()) {
            latred = new LatticeReductionImpl::Threaded3(p, cc);
        } else {
            // Phase 3: heuristic lattice reduction
            latred = new LatticeReductionImpl::Heuristic3(p, cc);
        }
    }

    assert(latred != nullptr);
    _is_configured = true;
}

void LatticeReduction::configure(const Matrix& M, const Matrix& U,
                                 const ComputationContext& cc) {
    configure(LatticeReductionParams(M, U), cc);
}

void LatticeReduction::solve() {
    assert(_is_configured);
    bool use_tasks = cc.is_threaded();

    if (use_tasks && omp_get_active_level() == 0) {
        #pragma omp parallel num_threads(cc.nthreads()) if (use_tasks)
        {
            #pragma omp single
            {
                latred->solve();
            }
        }
    } else {
        #pragma omp taskgroup
        {
            latred->solve();
        }
    }
}

void LatticeReduction::set_policy(unsigned int policy) {
    LatticeReduction::policy = policy;
}

}

#include "problems/lattice_reduction/base.h"

#include <cassert>
#include <cstring>
#include <sstream>

namespace flatter {
namespace LatticeReductionImpl {

const std::string Base::prob_name() {return "Lattice Reduction";}
const std::string Base::impl_name() {return "Base Implementation";}
const std::string Base::param_headers() {return "d:3 prec:1 alpha:-1 P:-1";}
std::string Base::get_param_values() {
    std::stringstream ss;
    double p = 0;
    MatrixData<mpz_t> dM = M.data<mpz_t>();
    for (unsigned int i = 0; i < dM.nrows(); i++) {
        for (unsigned int j = 0; j < dM.ncols(); j++) {
            double prec = mpz_sizeinbase(dM(i,j), 2);
            p = std::max(prec, p);
        }
    }
    double alpha = params.goal.get_max_drop() / 2;
    if (params.proved) {
        alpha -= HERMITE_BEST_SLOPE;
    } else {
        alpha -= BKZ_BEST_SLOPE;
    }
    ss << this->n << " " << p << " " << alpha;
    return ss.str(); 
}

Base::Base() :
    m(0),
    n(0),
    prec(0)
{}

Base::Base(const LatticeReductionParams& p,
           const ComputationContext& cc) {
    Base::configure(p, cc);
}

void Base::configure(const LatticeReductionParams& p,
                     const ComputationContext& cc) {
    Matrix M = p.B();
    Matrix U = p.U();
    
    assert(M.ncols() == U.nrows());
    assert(p.L.rank() == U.ncols());

    this->params = p;
    
    this->M = M;
    this->U = U;
    this->rhf = p.rhf();
    this->profile_offset = p.profile_offset;
    this->offset = p.offset;

    this->lvalid = p.lvalid;
    this->rvalid = p.rvalid;

    this->m = M.nrows();
    this->n = M.ncols();
    this->prec = M.prec();
    this->cc = cc;
}

void Base::_debug_mat(Matrix *mat) {
    unsigned int nrows = mat->nrows();
    unsigned int ncols = mat->ncols();

    MatrixData<mpz_t> dM = mat->data<mpz_t>();

    void (*free)(void *, size_t);
    mp_get_memory_functions (NULL, NULL, &free);

    // We're printing in FPLLL format (row notation)
    // but store the data in column notation, so print
    // the transpose.
    mon->debug("[");
    for (unsigned int i = 0; i < ncols; i++) {
        mon->debug("[");
        for (unsigned int j = 0; j < nrows; j++) {
            if (j) {
                mon->debug(" ");
            }

            char* elem = mpz_get_str(nullptr, 10, dM(j, i));
            mon->debug("%s", elem);
            free(elem, strlen(elem) + 1);
        }
        mon->debug("]\n");
    }
    mon->debug("]\n");


}

void Base::debug_matrix() {
    this->_debug_mat(&M);
    this->_debug_mat(&U);
}

}
}

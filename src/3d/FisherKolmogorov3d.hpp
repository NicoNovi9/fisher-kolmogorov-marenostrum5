#ifndef FISHER_KOLMOGORV_3D_HPP
#define FISHER_KOLMOGORV_3D_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_simplex_p.h>
// #include <deal.II/fe/fe_simplex_p.h>
// #include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>
// #include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <filesystem>
#include <fstream>
#include <iostream>

#include "DiffusionType.hpp"

// NOTE: E' applicato Eulero implicito direttamente. Per questo non c'è il
// parametro theta NOTE: Il dominio è già [-1, 1], non bisogna specificarlo
/* Ora farò riferimento al paper del progetto e la risoluzione del lab 8-2
A sezione 2.3 si deriva la weak formulation e la linearizza
    R_I --> corrisponde a R^(n+1)(u^(n+1)_h)    Residuo
    K_IJ --> corrisponde a a(u^(n+1)_h)         Derivata di Frechet di R
    N_i --> dovrebbe corrispondere con le basis functions
In nostro caso F = 0
Le boundary conditions sono Neumann omogenee

(mu_0 + mu_1*u^2) * grad(u) (scalar-vector prod)     --> D * grad(u)
(matrix-vector prod) Da aggiungere       + alpha * c * (1-c)

*/

using namespace dealii;

inline const std::string output_directory = "output";

struct SimulationParams {
    const std::string mesh_file_name;
    const unsigned int degree;
    const double T;
    const double deltat;
    const double d_axn;
    const double d_ext;
    const DiffusionType diffusion_type;
    const Point<3> seed_point;
    const double alpha;
    const double initial_seed;

    // Constructor to initialize from a key-value map
    SimulationParams(const std::map<std::string, std::string> &configMap);
};

// Function to read a CSV file and store key-value pairs in a map
SimulationParams readCSVData(const std::string &filename);

// Class representing the non-linear diffusion problem.
class FisherKolmogorov3D {
   public:
    // Physical dimension (1D, 2D, 3D)
    static constexpr unsigned int dim = 3;

    class FunctionD : public TensorFunction<2, dim, double> {
       public:
        FunctionD(const SimulationParams &params)
            : TensorFunction<2, dim>(),
              d_axn(params.d_axn),
              d_ext(params.d_ext),
              diffusion_type(params.diffusion_type),
              seed_point(params.seed_point) {}

        virtual Tensor<2, dim, double> value(
            const Point<dim> &p) const override {
            Tensor<2, dim> matrix;
            Tensor<1, dim> n_vector;
            Tensor<1, dim> n = compute_normal_vector(p);

            for (unsigned int i = 0; i < dim; i++) {
                matrix[i][i] = d_ext;
            }

            for (unsigned int i = 0; i < dim; ++i) {
                for (unsigned int j = 0; j < dim; ++j) {
                    matrix[i][j] += d_axn * n[i] * n[j];
                }
            }

            return matrix;
        }

       private:
        Tensor<1, dim> compute_normal_vector(const Point<dim> &p) const {
            Tensor<1, dim> n;
            Tensor<1, dim> t;  // Circumferential

            switch (diffusion_type) {
                case ISOTROPIC: {
                    for (unsigned int i = 0; i < dim; ++i) n[i] = 0.0;
                    break;
                }

                case RADIAL: {
                    n = p - seed_point;
                    double norm = n.norm();
                    if (norm > 1e-12) n /= norm;
                    break;
                }

                case CIRCUMFERENTIAL: {
                    t = p - seed_point;
                    double norm_t = t.norm();
                    if (norm_t > 1e-12) t /= norm_t;

                    n[0] = -t[1];
                    n[1] = t[0];
                    if (dim == 3) n[2] = 0;
                    double norm = n.norm();
                    if (norm > 1e-12) n /= norm;
                    break;
                }

                case AXON_BASED: {
                    // TODO
                }
            }
            return n;
        }

        const double d_axn;  // TODO: Trova valori che usano loro
        const double d_ext;  // TODO: Trova valori che usano loro
        const DiffusionType diffusion_type;
        const Point<dim> seed_point;
    };

    class FunctionAlpha : public Function<dim> {
       public:
        FunctionAlpha(const SimulationParams &params) : alpha(params.alpha) {}

        virtual double value(
            const Point<dim> & /*p*/,
            const unsigned int /*component*/ = 0) const override {
            return alpha;  // TODO: Trova valori che usano loro
        }

       private:
        const double alpha;
    };

    // Function for initial conditions.
    class FunctionU0 : public Function<dim> {
       public:
        FunctionU0(const SimulationParams &params) : 
            seed_point(params.seed_point),
            initial_seed(params.initial_seed) {}

        virtual double value(
            const Point<dim> &p,
            const unsigned int /*component*/ = 0) const override {
            // TODO: Pensa a come gestire initial_seeding_region tramite file di
            // inizializzazione
            if ((p[0] >= (seed_point[0] - 10.0) &&
                 p[0] <= (seed_point[0] + 10.0)) &&
                (p[1] >= (seed_point[1] - 10.0) &&
                 p[1] <= (seed_point[1] + 10.0)) &&
                (p[2] >= (seed_point[2] - 10.0) &&
                 p[2] <= (seed_point[2] + 10.0))) {
                return initial_seed;
            } else
                return 0.0;
        }

       private:
        const Point<dim> seed_point;
        const double initial_seed;
    };

    // Constructor. We provide the final time, time step Delta t and theta
    // method parameter as constructor arguments.
    FisherKolmogorov3D(SimulationParams params)
        : mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)),
          mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)),
          pcout(std::cout, mpi_rank == 0),
          simulation_params(params),
          D(params),
          alpha(params),
          u_0(params),
          T(params.T),
          mesh_file_name(params.mesh_file_name),
          r(params.degree),
          deltat(params.deltat),
          mesh(MPI_COMM_WORLD) {}

    // Initialization.
    void setup();

    // Solve the problem.
    void solve();

   protected:
    // Assemble the tangent problem.
    void assemble_system();

    // Solve the linear system associated to the tangent problem.
    void solve_linear_system();

    // Solve the problem for one time step using Newton's method.
    void solve_newton();

    // Output.
    void output(const unsigned int &time_step) const;

    // MPI parallel.
    // /////////////////////////////////////////////////////////////

    // Number of MPI processes.
    const unsigned int mpi_size;

    // This MPI process.
    const unsigned int mpi_rank;

    // Parallel output stream.
    ConditionalOStream pcout;

    // Problem definition.
    // ///////////////////////////////////////////////////////

    const SimulationParams simulation_params;

    // mu_0 coefficient.
    FunctionD D;

    // mu_1 coefficient.
    FunctionAlpha alpha;

    // Initial conditions.
    FunctionU0 u_0;

    // Current time.
    double time;

    // Final time.
    const double T;

    // Discretization.
    // ///////////////////////////////////////////////////////////

    // Path to the mesh file.
    const std::string mesh_file_name;

    // Polynomial degree.
    const unsigned int r;

    // Time step.
    const double deltat;

    // Mesh.
    parallel::fullydistributed::Triangulation<dim> mesh;

    // Finite element space.
    std::unique_ptr<FiniteElement<dim>> fe;

    // Quadrature formula.
    std::unique_ptr<Quadrature<dim>> quadrature;

    // DoF handler.
    DoFHandler<dim> dof_handler;

    // DoFs owned by current process.
    IndexSet locally_owned_dofs;

    // DoFs relevant to the current process (including ghost DoFs).
    IndexSet locally_relevant_dofs;

    // Jacobian matrix.
    TrilinosWrappers::SparseMatrix jacobian_matrix;

    // Residual vector.
    TrilinosWrappers::MPI::Vector residual_vector;

    // Increment of the solution between Newton iterations.
    TrilinosWrappers::MPI::Vector delta_owned;

    // System solution (without ghost elements).
    TrilinosWrappers::MPI::Vector solution_owned;

    // System solution (including ghost elements).
    TrilinosWrappers::MPI::Vector solution;

    // System solution at previous time step.
    TrilinosWrappers::MPI::Vector solution_old;
};

#endif /* FISHIER_KOLMOGORV_3D_HPP */
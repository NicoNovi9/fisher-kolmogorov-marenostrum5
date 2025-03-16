#include <fstream>
#include <iostream>
#include <mpi.h>
#include "FisherKolmogorov3d.hpp"

int main(int argc, char *argv[]) {
    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);
    
    int mpi_size;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    double total_start_time = MPI_Wtime();

    double setup_start_time = MPI_Wtime();
    SimulationParams simulation_params = readCSVData("../data/params.csv");
    FisherKolmogorov3D problem(simulation_params);
    problem.setup();
    double setup_end_time = MPI_Wtime();
    double setup_time = setup_end_time - setup_start_time;

    double solve_start_time = MPI_Wtime();
    problem.solve();
    double solve_end_time = MPI_Wtime();
    double solve_time = solve_end_time - solve_start_time;

    double total_end_time = MPI_Wtime();
    double total_time = total_end_time - total_start_time;

    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) {
        std::ofstream outfile;
        outfile.open("timing_results.csv", std::ios::app);
        outfile << mpi_size << "," 
                << setup_time << ","
                << solve_time << ","
                << total_time << "\n";
        outfile.close();

        std::cout << "Execution Time with " << mpi_size << " tasks:" << std::endl;
        std::cout << "  Setup Time: " << setup_time << " seconds" << std::endl;
        std::cout << "  Solve Time: " << solve_time << " seconds" << std::endl;
        std::cout << "  Total Time: " << total_time << " seconds" << std::endl;
    }

    return 0;
}



cmake_minimum_required(VERSION 3.16)

project(FisherKolmogorov)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)  # Oppure Debug se vuoi fare debug

find_package(deal.II REQUIRED HINTS /home/nct/nct01024/share/dealii)

# add_executable(simulation1d src/1d/simulation1d.cpp src/1d/FisherKolmogorov1d.cpp)
# deal_ii_setup_target(simulation1d)

add_executable(simulation3d src/3d/simulation3d.cpp src/3d/FisherKolmogorov3d.cpp)
deal_ii_setup_target(simulation3d)

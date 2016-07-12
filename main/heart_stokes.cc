#include "interfaces/stokes.h"
#include "pidomus.h"

#include "deal.II/base/numbers.h"

#include "Teuchos_CommandLineProcessor.hpp"
#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_oblackholestream.hpp"
#include "Teuchos_StandardCatchMacros.hpp"
#include "Teuchos_Version.hpp"

#include "mpi.h"
#include <iostream>
#include <string>

#include "elastic_problem.h"


void print_status(  std::string name,
                    std::string prm_file,
                    int dim,
                    // int spacedim,
                    int n_threads,
                    const MPI_Comm &comm,
                    bool check_prm);


int main (int argc, char *argv[])
{
  using namespace dealii;
  using namespace deal2lkit;

  Teuchos::CommandLineProcessor My_CLP;
  My_CLP.setDocString(
    "                ______     ___  ____   _ __________ \n"
    "                |  _  \\    |  \\/  | | | /  ______  \\ \n"
    "  ______________| | | |____| .  . | |_| \\ `--.___| | \n"
    " |__  __  ______| | | / _ \\| |\\/| | |_| |`--. \\____/ \n"
    "   | | | |      | |/ | (_) | |  | | |_| /\\__/ / \n"
    "   |_| |_|      |___/ \\___/\\_|  |_/\\___/\\____/ \n"
  );

  std::string pde_name="stokes";
  //My_CLP.setOption("pde", &pde_name, "name of the PDE (stokes, NS for navier stokes, or ALE for ALE navier stokes)");

  // int spacedim = 2;
  // My_CLP.setOption("spacedim", &spacedim, "dimensione of the whole space");

  int dim = 3;
  //My_CLP.setOption("dim", &dim, "dimension of the problem");

  int n_threads = 0;
  My_CLP.setOption("n_threads", &n_threads, "number of threads");

  std::string prm_file=pde_name+".prm";
  My_CLP.setOption("prm", &prm_file, "name of the parameter file");

  bool check_prm = false;
  My_CLP.setOption("check","no-check", &check_prm, "wait for a key press before starting the run");

  // My_CLP.recogniseAllOptions(true);
  My_CLP.throwExceptions(false);

  Teuchos::CommandLineProcessor::EParseCommandLineReturn
  parseReturn= My_CLP.parse( argc, argv );

  if ( parseReturn == Teuchos::CommandLineProcessor::PARSE_HELP_PRINTED )
    {
      return 0;
    }
  if ( parseReturn != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL   )
    {
      return 1; // Error!
    }

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv,
                                                      n_threads == 0 ? numbers::invalid_unsigned_int  : n_threads);

  const MPI_Comm &comm  = MPI_COMM_WORLD;

  Teuchos::oblackholestream blackhole;
  std::ostream &out = ( Utilities::MPI::this_mpi_process(comm) == 0 ? std::cout : blackhole );


  std::string string_pde_name="Stokes";


  My_CLP.printHelpMessage(argv[0], out);

  // deallog.depth_console (0);
  try
    {
      print_status(   string_pde_name+" Equations",
                      prm_file,
                      dim,
                      // spacedim,
                      n_threads,
                      comm,
                      check_prm);

      StokesInterface<3,3,LATrilinos> interface;
      piDoMUS<3,3,LATrilinos> stokes ("pi-DoMUS",interface);
      ParameterAcceptor::initialize(prm_file, pde_name+"_used.prm");


      stokes.lambdas.output_step = [&] (const double t,
                                        const typename LATrilinos::VectorType &/*sol*/,
                                        const typename LATrilinos::VectorType &/*sol_dot*/,
                                        const unsigned int /*step_number*/)
      {
//          auto &dof = interface.get_dof_handler();
        auto &tria = const_cast<parallel::distributed::Triangulation<3,3>&>
                     (interface.get_triangulation());
        double dt = interface.get_timestep();
        if (dt != dt)   // check if dt is NaN
          dt = 1;

        ElasticProblem<3> elastic_problem(t, dt);
        elastic_problem.run(tria);
      };

      stokes.run ();


      out << std::endl;
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}

void print_status(  std::string name,
                    std::string prm_file,
                    int dim,
                    // int spacedim,
                    int n_threads,
                    const MPI_Comm &comm,
                    bool check_prm)
{
  int numprocs  = Utilities::MPI::n_mpi_processes(comm);
  //  int myid      = Utilities::MPI::this_mpi_process(comm);

  Teuchos::oblackholestream blackhole;
  std::ostream &out = ( Utilities::MPI::this_mpi_process(comm) == 0 ? std::cout : blackhole );


  out << std::endl
      << "============================================================="
      << std::endl
      << "     Name:  " << name
      << std::endl
      << " Prm file:  " << prm_file
      << std::endl
      << "n threads:  " << n_threads
      << std::endl
      << "  process:  "  << getpid()
      << std::endl
      << " proc.tot:  "  << numprocs
      << std::endl
      // << " spacedim:  " << spacedim
      // << std::endl
      << "      dim:  " << dim
      // << std::endl
      // << "    codim:  " << spacedim-dim
      << std::endl;
  if (check_prm)
    {
      out<< "-------------------------------------------------------------"
         << std::endl;
      out << "Press [Enter] key to start...";
      if (std::cin.get() == '\n') {};
    }
  out << "============================================================="
      <<std::endl<<std::endl;

}

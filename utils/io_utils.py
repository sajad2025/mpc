#!/usr/bin/env python3

import os
import sys

class SuppressOutput:
    """
    A context manager that redirects stdout and stderr to devnull.
    This works for both Python's output and C-level output.
    """
    def __init__(self):
        self._stdout = None
        self._stderr = None
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for _ in range(2)]
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Save current file descriptors and redirect stdout/stderr to devnull
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)
        sys.stdout.flush()
        sys.stderr.flush()
        
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore file descriptors
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        
        # Close the null file descriptors
        for fd in self.null_fds:
            os.close(fd)
        
        # Close the saved file descriptors
        for fd in self.save_fds:
            os.close(fd)


def clean_acados_files():
    """
    Clean up Acados generated files.
    """
    print("Cleaning up Acados generated files...")
    
    # List of files to remove
    files_to_remove = [
        "acados_solver_kinematic_car.o",
        "libacados_ocp_solver_kinematic_car.dylib",
        "acados_ocp_kinematic_car.json"
    ]
    
    success = True
    try:
        # Use SuppressOutput to silence system commands
        with SuppressOutput():
            for file_name in files_to_remove:
                try:
                    # Check if file exists before attempting to remove
                    if os.path.exists(file_name):
                        os.remove(file_name)
                except Exception as e:
                    success = False
                    # This won't be printed during SuppressOutput, but we'll store the failure
    except Exception as e:
        print(f"Error with output suppression: {str(e)}")
        success = False
        
        # If SuppressOutput fails, try without it
        for file_name in files_to_remove:
            try:
                if os.path.exists(file_name):
                    os.remove(file_name)
                    print(f"Removed {file_name}")
            except Exception as e:
                print(f"Error removing {file_name}: {str(e)}")
                success = False
    
    if success:
        print("Cleanup completed successfully.")
    else:
        print("Cleanup completed with some errors.")


if __name__ == "__main__":
    # When run directly, perform cleanup
    clean_acados_files() 
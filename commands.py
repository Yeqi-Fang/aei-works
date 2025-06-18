import os
import subprocess
import shutil

# --- Configuration ---
COMMAND_PREFIX = "lal"
OUTPUT_FILE = "successful_help_output_python.txt"

def find_executables_in_path(prefix):
    """
    Finds all executable files with a given prefix within the current PATH.
    This function assumes the correct Conda environment is already activated.
    """
    print("--> Searching for commands starting with '{}' in the current PATH...".format(prefix))

    # Get the PATH environment variable from the currently running process.
    env_path_str = os.environ.get('PATH', '')
    if not env_path_str:
        print("ERROR: The PATH environment variable is not set.")
        return None

    executable_commands = set()
    # The PATH can have duplicate directories, so use a set to avoid re-scanning
    for directory in set(env_path_str.split(os.pathsep)):
        if not os.path.isdir(directory):
            continue
        try:
            for filename in os.listdir(directory):
                if filename.startswith(prefix):
                    full_path = os.path.join(directory, filename)
                    # Check if the file is actually an executable
                    if os.path.isfile(full_path) and os.access(full_path, os.X_OK):
                        executable_commands.add(filename)
        except OSError:
            # Ignore directories we can't access
            continue

    return sorted(list(executable_commands))

def process_commands(command_list, output_file):
    """
    Runs '--help' on each command and saves successful output to a file.
    """
    success_count = 0
    error_count = 0

    print(f"--> Processing {len(command_list)} commands. Output will be saved to '{output_file}'.")

    # Overwrite the file to start fresh
    with open(output_file, "w", encoding="utf-8") as f:
        import time
        
        for command in command_list:
            try:
                # set time out
                
                # Execute the command and capture its output
                result = subprocess.run(
                    [command, "--help"],
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='replace', # Avoids errors with non-standard characters
                    timeout=100,  # Set a timeout to avoid hanging on long-running commands,
                    # output to stdout
                )

                # if result.returncode == 0:
                    # SUCCESS
                print(f"SUCCESS: Capturing help for '{command}'")
                f.write("=================================================\n")
                f.write(f"--- Help output for: {command}\n")
                f.write("=================================================\n")
                f.write(result.stdout)
                f.write("\n\n")
                success_count += 1
                # else:
                #     # ERROR (non-zero exit code)
                #     print(f"SKIPPED: Command '{command}' produced an error or has no --help option.")
                #     error_count += 1

            except FileNotFoundError:
                print(f"SKIPPED: Command '{command}' could not be found during execution.")
                error_count += 1
            except Exception as e:
                print(f"SKIPPED: An unexpected error occurred with command '{command}': {e}")
                error_count += 1

    print("-------------------------------------------------")
    print("--> Done!")
    print(f"    Summary: {success_count} successful commands captured, {error_count} commands skipped.")

def main():
    """Main function to run the script."""
    # This script relies on being run from an activated Conda environment.
    commands = find_executables_in_path(COMMAND_PREFIX)

    if commands is not None:
        process_commands(commands, OUTPUT_FILE)

if __name__ == "__main__":
    main()
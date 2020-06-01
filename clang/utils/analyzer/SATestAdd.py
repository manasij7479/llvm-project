#!/usr/bin/env python

"""
Static Analyzer qualification infrastructure: adding a new project to
the Repository Directory.

 Add a new project for testing: build it and add to the Project Map file.
   Assumes it's being run from the Repository Directory.
   The project directory should be added inside the Repository Directory and
   have the same name as the project ID

 The project should use the following files for set up:
      - cleanup_run_static_analyzer.sh - prepare the build environment.
                                     Ex: make clean can be a part of it.
      - run_static_analyzer.cmd - a list of commands to run through scan-build.
                                     Each command should be on a separate line.
                                     Choose from: configure, make, xcodebuild
      - download_project.sh - download the project into the CachedSource/
                                     directory. For example, download a zip of
                                     the project source from GitHub, unzip it,
                                     and rename the unzipped directory to
                                     'CachedSource'. This script is not called
                                     when 'CachedSource' is already present,
                                     so an alternative is to check the
                                     'CachedSource' directory into the
                                     repository directly.
      - CachedSource/ - An optional directory containing the source of the
                                     project being analyzed. If present,
                                     download_project.sh will not be called.
      - changes_for_analyzer.patch - An optional patch file for any local
                                     changes
                                     (e.g., to adapt to newer version of clang)
                                     that should be applied to CachedSource
                                     before analysis. To construct this patch,
                                     run the download script to download
                                     the project to CachedSource, copy the
                                     CachedSource to another directory (for
                                     example, PatchedSource) and make any
                                     needed modifications to the copied
                                     source.
                                     Then run:
                                          diff -ur CachedSource PatchedSource \
                                              > changes_for_analyzer.patch
"""
import SATestBuild
from ProjectMap import ProjectMap, ProjectInfo

import os
import sys


def add_new_project(name: str, build_mode: int):
    """
    Add a new project for testing: build it and add to the Project Map file.
    :param name: is a short string used to identify a project.
    """

    project_info = ProjectInfo(name, build_mode)
    test_info = SATestBuild.TestInfo(project_info,
                                     is_reference_build=True)
    tester = SATestBuild.ProjectTester(test_info)

    project_dir = tester.get_project_dir()
    if not os.path.exists(project_dir):
        print(f"Error: Project directory is missing: {project_dir}")
        sys.exit(-1)

    # Build the project.
    tester.test()

    # Add the project name to the project map.
    project_map = ProjectMap(should_exist=False)

    if is_existing_project(project_map, name):
        print(f"Warning: Project with name '{name}' already exists.",
              file=sys.stdout)
        print("Reference output has been regenerated.", file=sys.stdout)
    else:
        project_map.projects.append(project_info)
        project_map.save()


def is_existing_project(project_map: ProjectMap, project_name: str) -> bool:
    return any(existing_project.name == project_name
               for existing_project in project_map.projects)


# TODO: Use argparse
# TODO: Add an option not to build.
# TODO: Set the path to the Repository directory.
if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Add a new project for testing to the analyzer"
              "\nUsage: ", sys.argv[0],
              "project_ID <mode>\n"
              "mode: 0 for single file project, "
              "1 for scan_build, "
              "2 for single file c++11 project", file=sys.stderr)
        sys.exit(-1)

    build_mode = 1
    if len(sys.argv) >= 3:
        build_mode = int(sys.argv[2])

    assert((build_mode == 0) | (build_mode == 1) | (build_mode == 2))

    add_new_project(sys.argv[1], build_mode)

#!/usr/bin/env bash

# Checking VTC/ALICE dependencies

ERROR=false

if { git --version; } >/dev/null 2>&1; then
  echo "git found."
else
  ERROR=true
  echo "ERROR: git not found."
fi;

if { sox --version; } >/dev/null 2>&1; then
  echo "sox found."
else
  ERROR=true
  echo "ERROR: sox not found."
fi;

if { conda --version; } >/dev/null 2>&1; then
  echo "conda found."
else
  ERROR=true
  echo "ERROR: conda not found."
fi;

if { python --version; } >/dev/null 2>&1; then
  echo "python found."
else
  ERROR=true
  echo "ERROR: python not found."
fi;

# Checking VTC conda env.
if { conda env list | grep 'pyannote'; } >/dev/null 2>&1; then
  echo "pyannote conda environment found."
else
  ERROR=true
  echo "ERROR: pyannote conda environment not found. Please follow installation instructions for the VTC or come to the 12:00 to 01:00 debugging session"
fi;

# Checking ALICE conda env.
if { conda env list | grep 'ALICE'; } >/dev/null 2>&1; then
  echo "ALICE conda environment found."
else
  ERROR=true
  echo "ERROR: ALICE conda environment not found. Please follow installation instructions for ALICE or come to the 12:00 to 01:00 debugging session"
fi;

# Checking ChildProject conda env.
if { conda env list | grep 'childproject'; } >/dev/null 2>&1; then
  echo "childproject conda environment found."
else
  ERROR=true
  echo "ERROR: childproject conda environment not found. Please follow installation instructions for ChildProject or come to the 12:00 to 01:00 debugging session"
fi;

if [ $ERROR == true ]; then
  echo "You have some errors to fix!"
else
echo "%    ██████                        ██████"
echo "%  ██████████  ████████████████  ██████████"
echo "%██████████████                ██████████████"
echo "%████████                            ████████"
echo "%██████                                ██████"
echo "%  ██                                    ██"
echo "%  ██                                    ██"
echo "%██        ██████            ██████        ██"
echo "%██      ██████████        ██████████      ██"
echo "%██    ████████  ██        ██  ████████    ██"
echo "%██    ████████  ██        ██  ████████    ██"
echo "%██    ██████████            ██████████    ██"
echo "%██      ██████      ████      ██████      ██"
echo "%  ██                ████                ██"
echo "%  ████████  ▒▒▒▒▒▒        ▒▒▒▒▒▒  ████████"
echo "%████████████▒▒▒▒▒▒▒▒    ▒▒▒▒▒▒▒▒████████████"
echo "%██████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒██████████████"
echo "%██████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒██████████████"
echo "%██████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒██████████████"
echo "%  ████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒████████████"
echo "%    ████████  ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒  ████████"
echo "%                ▒▒▒▒▒▒▒▒▒▒▒▒"
echo "%                  ▒▒▒▒▒▒▒▒"
echo "%                    ▒▒▒▒"
  echo "Congrats you're ready for the tutorial!"
fi;

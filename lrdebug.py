'''
This module is intended to help debugging long running processes in python. It
tries to be as non-intrusive as possible by not introducing anything except its
name into the name space and not importing anything except "os" and "sys" until
it is needed. Even then it will at the most import "signal". Should debugging
become necessary it will also import "pdb", the debugger. It also imports
"traceback" but only when the main program is no longer running.

This module provides two "behaviours":

One is that it makes the program catch the "USR1" signal and then put itself in
debugging mode. The program is then in a suspended state where variables can be
checked and even changed, the code can be executed line by line, conditional
break-points be set and cleared et.c. To continue one can simply give the
command "c" (provided that no interfering break-points have been set) and the
program will resume, possibly with its state altered.

Another is that it puts the process in post-mortem debugging mode should any
uncaught exception cause the main program to exit. This makes it possible to
thoroughly inspect the state of the program and what caused it to raise an
exception.

To use, import the module thus:

import lrdebug

The above described behaviours will be available on assigning 1 to any
combination of the two environment variables:

USR_DEBUGGING # for responding to a signal
PM_DEBUGGING  # for post-mortem debugging

To remove either behaviour, just unset the corresponding variable. Note that
this has no effect on a running process (which would then have to change its
own environment). In other words, setting the variables on the commandline or
exporting them to the environment in some other way turns on these debugging
modes, but changing them internally in the python code doesn't have any effect.
'''

import os
import sys


def print_process_status():
    print(f'Running:\n\t{sys.argv[0]}\n')
    print('Commandline Parameters:')
    for i in range(1, len(sys.argv)):
        print(f'\t{i}: "{sys.argv[i]}"')
    print()
    pid = os.getpid()
    print(f'Process id of main process is:\n\t{pid}\n')
    print(f'\nTo halt and debug a process (in its own terminal) send signal USR1 (kill -10 <pid>, e.g. kill -10 {pid})\n')


def usr_debug(sig, frame):
    import pdb
    if sys.version_info.major > 3 or (sys.version_info.major == 3 and sys.version_info.minor > 7):
        print(f'Received signal {sig} ({signal.strsignal(sig)}) in responce to which debug mode will be attempted\n')
    else:
        print('Received a signal in responce to which debug mode will be attempted\n')
    print('Dropping to debugger\n')
    print('User debugger initiated (unset USR_DEBUGGING to disable)\n')
    pdb.Pdb().set_trace(frame)


def pm_debug(e):
    import pdb
    import traceback
    traceback.print_tb(e.__traceback__)
    print(f'Received {type(e).__name__}: {e}\n')
    print('Dropping to debugger\n')
    print('Postmortem debugger initiated (unset PM_DEBUGGING to disable)\n')
    pdb.post_mortem(e.__traceback__)


if 'PM_DEBUGGING' in os.environ and os.environ['PM_DEBUGGING']:
    sys.excepthook = lambda e_type, value, traceback: pm_debug(value)


if 'USR_DEBUGGING' in os.environ and os.environ['USR_DEBUGGING']:    
    import signal
    signal.signal(signal.SIGUSR1, usr_debug)
    print_process_status()


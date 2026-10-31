# v57r1 original shared-MMA screen

Status: `research_rejected`. Source `cf6efa174aaa5d3c035a937ce7daa65ff1bf4fc0`; both research source files verified byte for byte against Git. This is not model training or a task score. Production v45 is unchanged.

Error: `PTXASError: PTXAS error: Internal Triton PTX codegen error
`ptxas` stderr:
ptxas /tmp/tmpq3i1o7fa.ptx, line 272; error   : Arguments mismatch for instruction 'mma'
ptxas /tmp/tmpq3i1o7fa.ptx, line 274; error   : Arguments mismatch for instruction 'mma'
ptxas /tmp/tmpq3i1o7fa.ptx, line 276; error   : Arguments mismatch for instruction 'mma'
ptxas /tmp/tmpq3i1o7fa.ptx, line 278; error   : Arguments mismatch for instruction 'mma'
ptxas /tmp/tmpq3i1o7fa.ptx, line 280; error   : Arguments mismatch for instruction 'mma'
ptxas /tmp/tmpq3i1o7fa.ptx, line 282; error   : Arguments mismatch for instruction 'mma'
ptxas /tmp/tmpq3i1o7fa.ptx, line 284; error   : Arguments mismatch for instruction 'mma'
ptxas /tmp/tmpq3i1o7fa.ptx, line 286; error   : Arguments mismatch for instruction 'mma'
ptxas /tmp/tmpq3i1o7fa.ptx, line 302; error   : Arguments mismatch for instruction 'mma'
ptxas /tmp/tmpq3i1o7fa.ptx, line 304; error   : Arguments mismatch for instruction 'mma'
ptxas /tmp/tmpq3i1o7fa.ptx, line 306; error   : Arguments mismatch for instruction 'mma'
ptxas /tmp/tmpq3i1o7fa.ptx, line 308; error   : Arguments mismatch for instruction 'mma'
ptxas /tmp/tmpq3i1o7fa.ptx, line 310; error   : Arguments mismatch for instruction 'mma'
ptxas /tmp/tmpq3i1o7fa.ptx, line 312; error   : Arguments mismatch for instruction 'mma'
ptxas /tmp/tmpq3i1o7fa.ptx, line 314; error   : Arguments mismatch for instruction 'mma'
ptxas /tmp/tmpq3i1o7fa.ptx, line 316; error   : Arguments mismatch for instruction 'mma'
ptxas /tmp/tmpq3i1o7fa.ptx, line 332; error   : Arguments mismatch for instruction 'mma'
ptxas /tmp/tmpq3i1o7fa.ptx, line 334; error   : Arguments mismatch for instruction 'mma'
ptxas /tmp/tmpq3i1o7fa.ptx, line 336; error   : Arguments mismatch for instruction 'mma'
ptxas /tmp/tmpq3i1o7fa.ptx, line 338; error   : Arguments mismatch for instruction 'mma'
ptxas /tmp/tmpq3i1o7fa.ptx, line 340; error   : Arguments mismatch for instruction 'mma'
ptxas /tmp/tmpq3i1o7fa.ptx, line 342; error   : Arguments mismatch for instruction 'mma'
ptxas /tmp/tmpq3i1o7fa.ptx, line 344; error   : Arguments mismatch for instruction 'mma'
ptxas /tmp/tmpq3i1o7fa.ptx, line 346; error   : Arguments mismatch for instruction 'mma'
ptxas /tmp/tmpq3i1o7fa.ptx, line 362; error   : Arguments mismatch for instruction 'mma'
ptxas /tmp/tmpq3i1o7fa.ptx, line 364; error   : Arguments mismatch for instruction 'mma'
ptxas /tmp/tmpq3i1o7fa.ptx, line 366; error   : Arguments mismatch for instruction 'mma'
ptxas /tmp/tmpq3i1o7fa.ptx, line 368; error   : Arguments mismatch for instruction 'mma'
ptxas /tmp/tmpq3i1o7fa.ptx, line 370; error   : Arguments mismatch for instruction 'mma'
ptxas /tmp/tmpq3i1o7fa.ptx, line 372; error   : Arguments mismatch for instruction 'mma'
ptxas /tmp/tmpq3i1o7fa.ptx, line 374; error   : Arguments mismatch for instruction 'mma'
ptxas /tmp/tmpq3i1o7fa.ptx, line 376; error   : Arguments mismatch for instruction 'mma'
ptxas fatal   : Ptx assembly aborted due to errors

Repro command: /usr/local/lib/python3.13/dist-packages/triton/backends/nvidia/bin/ptxas -lineinfo --fmad=false -v --gpu-name=sm_75 /tmp/tmpq3i1o7fa.ptx -o /tmp/tmpq3i1o7fa.ptx.o
`

See screen.json for unaltered numerical and timing evidence. No original ten-check admission or <=27-second claim is made for this isolated screen.

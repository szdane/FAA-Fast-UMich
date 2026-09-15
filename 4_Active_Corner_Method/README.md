# Active Corners Certificate Generation

As described in the associated paper, this library automatically constructs proof certificates for instances of the active corner method. To run these examples, you will need to have Python, SymPy, PVS 8 (PVS 7 may not work), and NASALib installed. A variety of examples (`example_1.py` through `example_7.py`) are included to illustrate key examples as described in our paper. Our case study is also included as a PVS file: `dl_case_study.pvs`.

Each example defines a polygon, trajectory function, and domain. A proof certificate is automatically generated and saved to a PVS file with the same name as the example; that is, `example_1.pvs`, `example_2.pvs`, and so on.

Running an example is simple: run the following command from the terminal where `N` is the number of the example you want to run.

```
$ python3 example_N.py
```

This will generate a PVS file with the same name as the example, and print a message to the console indicating that the proof has been generated and saved to the file.

You can check a certificate in two ways:

- From the terminal, run `proveit example_N.pvs` where `N` is the number of the example whose Python file you ran earlier. The example will run for a while (in our experience on an M1 MacBook Pro, 30 seconds or so), and print a proof summary to the console.
- You can open the PVS file directly and install Prooflite scripts by placing your cursor over them, typing `M-x install-prooflite-script`. This is more tedious, but useful if you want to see the proofs as they progress.

Two potential errors you may encounter:

- If `proveit` fails with path errors, you may need to move the certificate to a directory without spaces in its path and run `proveit` from there.
- To avoid errors for duplicate theory names, move certificates into their own directory or remove PVS artifacts before checking another certificate. PVS caches theories and all active corner certificates are declared with the same theory name `active_corner_certificate`.

Only one example is expected to not have all formulas successfully prove: `example_7`. This is due to a limitation of the PVS `deriv` command, which fails to reason about the non-expanded form of the quadratic function. Example 6 describes the same trajectory as Example 7, but the trajectory is expanded, so the last lemma succeeds since `deriv` works on that function. See below for example output from running `example_6.pvs` and `example_7.pvs`.

```
Processing example_6.pvs. Writing output to file /path/to/certificate/example_6.summary

 Proof summary for theory active_corner_certificate
    Theory totals: 48 formulas, 48 attempted, 48 succeeded (0.04 s)

Grand Totals: 48 proofs, 48 attempted, 48 succeeded (0.04 s)
```

```
Processing example_7.pvs. Writing output to file /path/to/certificate/example_7.summary

 Proof summary for theory active_corner_certificate
    full_domain_soundness_lemma...........unfinished          [SHOSTAK](0.02 s)
    Theory totals: 48 formulas, 48 attempted, 47 succeeded (0.03 s)

Grand Totals: 48 proofs, 48 attempted, 47 succeeded (0.03 s)
*** Warning: Missed 1 formula.
```

## Case Studies

Our two case studies from the paper are included:

- `dl_case_study.pvs`, along with associated proof and summary files.
- `strategic.pvs`, along with associated proof and summary files.

These can be run with `proveit` or explored in the PVS editor.

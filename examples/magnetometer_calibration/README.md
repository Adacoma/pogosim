# Magnetometer calibration example

To use a reference csv from experimental data (pogobot dump + video):
```shell
mkdir -p tmp_magneto
# Copy the UART and camera files into the tmp_magneto directory

./scripts/make_magnetometer_reference.py real --uart tmp_magneto/uart_angles.txt --camera tmp_magneto/video_angles.txt -o tmp_magneto/magnetometer_reference_real.csv
mv tmp_magneto/magnetometer_reference_real.csv conf/example_magnetometer_reference.csv
```

To launch the simulation to reproduce the experimental magnetometer elliptic bias, and make plots:
```shell
./examples/magnetometer_calibration/magnetometer_calibration -c conf/magnetometer.yaml > tmp_magneto/dump_pogosim.txt
cp frames/data.feather tmp_magneto/data.feather
./examples/magnetometer_calibration/plot_magnetometer_simulation_report.py --dump tmp_magneto/dump_pogosim.txt --data tmp_magneto/data.feather --robot-id 0 -o tmp_magneto/sim_report
```

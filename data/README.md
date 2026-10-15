# Raw deployment data

`data.csv.gz`: 947,682 records logged by the greenhouse monitoring system
between 18 April and 22 May 2022 (median interval ~1 s, irregular), distributed
gzip-compressed (5.6 MB compressed, 62 MB uncompressed). pandas reads it
directly: `pd.read_csv("data/data.csv.gz", encoding="latin-1")`.

Columns: Id, humidity (%), temperature (°C), humiditysol (soil moisture, %),
temperaturesol (soil temperature, °C), co2 (ppm), lumière (light, binary), date.

The file is uncurated: it contains the naturally occurring faults analyzed in
the manuscript (transmission dropouts, out-of-range values, stuck-at runs, a
zero-variance channel, and a suspected CO₂ calibration bias). Do not clean it
before running the pipeline; the faults are the object of study.

Provenance: tomato greenhouse, Casablanca, Morocco. The previous deployment
description (doi:10.23939/mmc2023.02.524) documents the DHT22 (air humidity and
temperature), DS18B20 (soil temperature), soil-moisture sensor v1.2, digital
LDR (light), and the ESP32 MQTT client logging over Wi-Fi to a Raspberry Pi 4
broker. The CO₂ stream is present in the raw analyzed database, but its sensor
model, interface, and specifications are not recorded in that source (n/r).

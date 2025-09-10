package org.fog.test.perfeval;

import org.fog.entities.Sensor;
import org.fog.entities.Tuple;
import org.fog.utils.distribution.Distribution;
import org.fog.utils.FogEvents;
import org.fog.utils.FogUtils;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.logging.Logger;

import org.cloudbus.cloudsim.UtilizationModelFull;
import org.cloudbus.cloudsim.core.CloudSim;

/**
 * Custom Sensor: Generates synthetic data, can be stopped on fault.
 */
public class CustomSensor extends Sensor {
    private static final Logger LOGGER = Logger.getLogger(CustomSensor.class.getName());
    private Random random = new Random();
    private static Map<String, Boolean> stoppedSensors = new HashMap<>(); // Shared stop flags

    public CustomSensor(String name, String tupleType, int userId, String appId, Distribution dist) {
        super(name, tupleType, userId, appId, dist);
    }

    @Override
    public void transmit() {
        if (stoppedSensors.getOrDefault(getName(), false)) {
            LOGGER.warning("Sensor " + getName() + " is stopped due to detected fault. Skipping transmit at time " + CloudSim.clock());
            return;
        }

        boolean isFault = random.nextDouble() < 0.2;
        List<Double> vibration = new ArrayList<>();
        for (int i = 0; i < 100; i++) {
            double val = isFault ? 2 * Math.sin(2 * i * 0.1) + random.nextGaussian() * 0.5 : Math.sin(i * 0.1) + random.nextGaussian() * 0.2;
            if (random.nextDouble() < 0.05) val = Double.NaN;
            vibration.add(val);
        }
        double temp = isFault ? random.nextGaussian() * 5 + 70 : random.nextGaussian() * 3 + 50;
        if (random.nextDouble() < 0.05) temp = Double.NaN;
        double voltage = isFault ? random.nextGaussian() * 10 + 250 : random.nextGaussian() * 5 + 220;
        if (random.nextDouble() < 0.05) voltage = Double.NaN;

        // Simulate mobility: Random latency change
        setLatency(1.0 + random.nextDouble() * 0.5); // Vary latency for realism

        Map<String, Object> payload = new HashMap<>();
        payload.put("vibration", vibration);
        payload.put("temp", temp);
        payload.put("voltage", voltage);
        payload.put("true_fault", isFault ? 1 : 0);
        String name = getName();
        int machineId = Integer.parseInt(name.split("-")[1]);
        payload.put("machine_id", machineId);

        DataTuple tuple = new DataTuple(getAppId(), FogUtils.generateTupleId(), Tuple.UP, 1000, 1, 800, 0,
            new UtilizationModelFull(), new UtilizationModelFull(), new UtilizationModelFull());
        tuple.setUserId(getUserId());  // ADD THIS LINE: Set userId to match the sensor's userId
        tuple.setTupleType("SENSOR");
        tuple.setPayload(payload);

        List<String> destModules = new ArrayList<>();
        destModules.add("Preprocess");
        tuple.setDestModuleNames(destModules);
        send(getGatewayDeviceId(), 0, FogEvents.TUPLE_ARRIVAL, tuple);
        LOGGER.info("Sensor " + getName() + " transmitted data at time " + CloudSim.clock() + " (true_fault=" + (isFault ? 1 : 0) + ")");
    }

    public static void stopSensor(String sensorName) {
        stoppedSensors.put(sensorName, true);
    }
}
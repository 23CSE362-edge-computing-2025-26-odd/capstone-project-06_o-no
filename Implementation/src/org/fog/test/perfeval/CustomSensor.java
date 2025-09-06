package org.fog.test.perfeval;

import org.fog.entities.Sensor;
import org.fog.entities.Tuple;
import org.fog.utils.distribution.Distribution;
import org.fog.utils.FogEvents;
import org.fog.utils.FogUtils;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.HashMap;
import java.util.Map;

public class CustomSensor extends Sensor {
    private Random random = new Random();

    public CustomSensor(String name, String tupleType, int userId, String appId, Distribution dist) {
        super(name, tupleType, userId, appId, dist);
    }

    @Override
	public void transmit() {
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

        Map<String, Object> payload = new HashMap<>();
        payload.put("vibration", vibration);
        payload.put("temp", temp);
        payload.put("voltage", voltage);
        payload.put("true_fault", isFault ? 1 : 0);  

        DataTuple tuple = new DataTuple(getAppId(), FogUtils.generateTupleId(), Tuple.UP, 1000, 1, 800, 0, new org.cloudbus.cloudsim.UtilizationModelFull(), new org.cloudbus.cloudsim.UtilizationModelFull(), new org.cloudbus.cloudsim.UtilizationModelFull());
        tuple.setTupleType("SENSOR");
        tuple.setPayload(payload);
        send(getGatewayDeviceId(), 0, FogEvents.TUPLE_ARRIVAL, tuple);
    }
}
package org.fog.test.perfeval;

import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.core.SimEvent;
import org.fog.entities.Actuator;
import org.fog.entities.Tuple;
import java.util.logging.Logger;

/**
 * Custom Actuator: Processes STOP tuples and stops corresponding sensor.
 */
public class CustomActuator extends Actuator {
    private static final Logger LOGGER = Logger.getLogger(CustomActuator.class.getName());

    public CustomActuator(String name, int userId, String appId, String actuatorType) {
        super(name, userId, appId, actuatorType);
    }

    @Override
    protected void processTupleArrival(SimEvent ev) {
        super.processTupleArrival(ev);
        Tuple tuple = (Tuple) ev.getData();
        if (tuple.getTupleType().equals(this.getActuatorType())) {
            String sensorName = getName().replace("actuator", "sensor");
            CustomSensor.stopSensor(sensorName);
            MetricsCollector.incrementStoppedMachines();
            LOGGER.info("Actuator " + getName() + " received STOP tuple at time " + CloudSim.clock() + " (fault detected, stopped sensor: " + sensorName + ")");
        }
    }
}
package org.fog.test.perfeval;

import org.apache.commons.math3.util.Pair;
import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.core.SimEntity;
import org.cloudbus.cloudsim.core.SimEvent;
import org.fog.entities.Actuator;
import org.fog.entities.FogDevice;
import org.fog.entities.Sensor;
import org.fog.placement.Controller;
import org.fog.utils.FogUtils;

import java.util.logging.Logger;

/**
 * Monitor Entity: Periodically checks load and dynamically adds edges.
 */
public class Monitor extends SimEntity {
    private static final Logger LOGGER = Logger.getLogger(Monitor.class.getName());
    private double interval;
    private double loadThreshold;
    private Controller controller;
    private int nextEdgeId;

    public Monitor(String name, double interval, double loadThreshold, Controller controller) {
        super(name);
        this.interval = interval;
        this.loadThreshold = loadThreshold;
        this.controller = controller;
        this.nextEdgeId = IntelliPdM.initialNumEdges; // Start from next ID
    }

    @Override
    public void startEntity() {
        sendNow(getId(), 0); // Start monitoring
    }

    @Override
    public void processEvent(SimEvent ev) {
        switch (ev.getTag()) {
            case 0: // Monitor event
                checkAndScale();
                send(getId(), interval, 0); // Schedule next check
                break;
        }
    }

    private void checkAndScale() {
        for (FogDevice fd : new java.util.ArrayList<>(IntelliPdM.fogDevices)) {
            if (fd.getName().startsWith("edge")) {
                double utilization = fd.getHost().getUtilizationOfCpu(); // Check CPU load
                if (utilization > loadThreshold) {
                    LOGGER.warning("Edge " + fd.getName() + " overloaded (util=" + utilization + ") at time " + CloudSim.clock() + ". Adding new edge.");
                    FogDevice cloud = IntelliPdM.fogDevices.get(0); // Assume index 0 is cloud
                    IntelliPdM.addEdgeDevice(cloud, nextEdgeId++, "IntelliPdM", FogUtils.USER_ID);
                    // Migrate some sensors (e.g., half)
                    migrateSensors(fd);
                }
            }
        }
        MetricsCollector.updateNetworkUsage(); // Update metrics
    }

    private void migrateSensors(FogDevice oldEdge) {
        int migrated = 0;
        FogDevice newEdge = IntelliPdM.fogDevices.get(IntelliPdM.fogDevices.size() - 1); // Last added
        for (Sensor sensor : IntelliPdM.sensors) {
            if (sensor.getGatewayDeviceId() == oldEdge.getId() && migrated < IntelliPdM.numMachines / 2) {
                sensor.setGatewayDeviceId(newEdge.getId());
                newEdge.getChildToLatencyMap().put(sensor.getId(), 1.0);
                oldEdge.getChildToLatencyMap().remove(sensor.getId());
                // Migrate corresponding actuator
                String actName = sensor.getName().replace("sensor", "actuator");
                for (Actuator act : IntelliPdM.actuators) {
                    if (act.getName().equals(actName)) {
                        act.setGatewayDeviceId(newEdge.getId());
                        newEdge.getChildToLatencyMap().put(act.getId(), 1.0);
                        oldEdge.getChildToLatencyMap().remove(act.getId());

                        // Remove from old edge's associatedActuatorIds
                        Pair<Integer, Double> toRemove = null;
                        for (Pair<Integer, Double> p : oldEdge.getAssociatedActuatorIds()) {
                            if (p.getKey() == act.getId()) {
                                toRemove = p;
                                break;
                            }
                        }
                        if (toRemove != null) {
                            oldEdge.getAssociatedActuatorIds().remove(toRemove);
                        }

                        // Add to new edge's associatedActuatorIds
                        newEdge.getAssociatedActuatorIds().add(new Pair<>(act.getId(), 1.0));

                        break;
                    }
                }
                migrated++;
                LOGGER.info("Migrated " + sensor.getName() + " to new edge " + newEdge.getName());
            }
        }
    }

    @Override
    public void shutdownEntity() {
        // No-op
    }
}
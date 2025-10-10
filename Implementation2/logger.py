
class SystemLogger:
    def __init__(self, metrics):
        self.metrics = metrics

    def log_event(self, env_time, event_type, details):
        log_entry = {'timestamp': env_time, 'type': event_type, 'details': details}
        self.metrics.setdefault('logs', []).append(log_entry)

        if event_type in ['FAULT_DETECTED', 'SYSTEM_ALERT', 'OFFLOAD_DECISION']:
            try:
                print(f"[{env_time:.1f}] {event_type}: {details}")
            except Exception:
                pass

    def log_task_completion(self, env_time, location, task_id, latency, energy, accuracy, fault_detected, infer_ms=None, **kwargs):
        self.metrics.setdefault('timeline', [])

        entry = {
            'time': env_time,
            'task_id': task_id,
            'location': location,
            'latency': latency,
            'energy': energy,
            'accuracy': accuracy,
            'fault_detected': bool(fault_detected)
        }

        if infer_ms is not None:
            entry['infer_ms'] = float(infer_ms)
        for k, v in kwargs.items():
            if k in entry:
                entry[f"extra_{k}"] = v
            else:
                entry[k] = v

        self.metrics['timeline'].append(entry)
        try:
            if location == 'edge' and 'edge' in self.metrics:
                self.metrics['edge'].setdefault('latency', []).append(latency)
                self.metrics['edge'].setdefault('energy', []).append(energy)
                self.metrics['edge'].setdefault('accuracy', []).append(accuracy)
                if infer_ms is not None:
                    self.metrics['edge'].setdefault('processing_times', []).append(infer_ms)
            elif location == 'cloud' and 'cloud' in self.metrics:
                self.metrics['cloud'].setdefault('latency', []).append(latency)
                self.metrics['cloud'].setdefault('energy', []).append(energy)
                self.metrics['cloud'].setdefault('accuracy', []).append(accuracy)
                if infer_ms is not None:
                    self.metrics['cloud'].setdefault('processing_times', []).append(infer_ms)
        except Exception:
            pass
        if fault_detected:
            try:
                print(f"[{env_time:.1f}] TASK {task_id} completed at {location} — fault_detected={fault_detected}, latency={latency:.2f}, infer_ms={infer_ms if infer_ms is not None else 'N/A'}")
            except Exception:
                pass

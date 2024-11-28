import psutil
import GPUtil
from typing import Dict

class MetricsCollector:
    @staticmethod
    def collect_cpu_metrics() -> Dict:
        cpu_percent = psutil.cpu_percent(interval=0.1, percpu=True)
        return {
            "cpu_utilization": cpu_percent,
            "cpu_average": sum(cpu_percent) / len(cpu_percent)
        }
    
    @staticmethod
    def collect_gpu_metrics() -> Dict:
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Get first GPU
                return {
                    "gpu_utilization": gpu.load * 100,
                    "gpu_memory_used": gpu.memoryUsed,
                    "gpu_memory_total": gpu.memoryTotal
                }
        except:
            pass
        return {}
    
    @staticmethod
    def collect_memory_metrics() -> Dict:
        memory = psutil.virtual_memory()
        return {
            "memory_used": memory.used / (1024**3),  # GB
            "memory_available": memory.available / (1024**3),  # GB
            "memory_percent": memory.percent
        }
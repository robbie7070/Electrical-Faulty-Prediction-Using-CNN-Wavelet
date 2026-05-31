# predict_menu.py
"""
Standalone CLI prediction system for real-time fault detection
Tanzania 230V/400V Distribution Network
"""

import numpy as np
import sys
import config
from predictor import Predictor


class PredictionMenu:
    """Interactive CLI for real-time fault prediction."""
    
    def __init__(self):
        self.predictor = Predictor()
        self.buffer = {'voltage': [], 'current': [], 'temperature': []}
        self.window_size = config.WINDOW_SIZE
        self.running = True
        self.measurement_count = 0
        self.prediction_count = 0
        self.fault_history = []
    
    # =========================================================================
    # HELPERS
    # =========================================================================
    
    def _fault_info(self, fault_type):
        info = {
            "Normal":                  ("✅", "NORMAL",              "🟢"),
            "Overload":                ("🟠", "WARNING - OVERLOAD",   "🟠"),
            "Transformer_Overheating": ("🔴", "CRITICAL - OVERHEAT",   "🔴"),
            "Loose_Connection":        ("🟡", "WARNING - LOOSE",      "🟡"),
        }
        return info.get(fault_type, ("⚠️", "ALERT", "🟡"))
    
    def _recommendations(self, fault_type):
        recs = {
            "Normal":                  ["✅ System operating normally", "Continue routine monitoring"],
            "Overload":                ["Check for excessive load", "Inspect for short circuits", "Verify protection"],
            "Transformer_Overheating": ["Inspect cooling systems", "Check oil temperature", "Reduce loading"],
            "Loose_Connection":        ["Check terminal connections", "Inspect for fluctuations", "Tighten connections"],
        }
        return recs.get(fault_type, ["Manual inspection recommended"])
    
    def _rpad(self, text, width):
        return text + " " * max(0, width - len(text))
    
    def _box(self, text, width=54):
        return f"║  {self._rpad(text, width - 4)}║"
    
    def _sep(self, width=54):
        return f"╠{'═'*width}╣"
    
    # =========================================================================
    # INPUT
    # =========================================================================
    
    def _input_measurement(self):
        """Get one measurement. Returns True/False/'stop'/'exit'/None."""
        try:
            v = input("   Voltage (V): ").strip().lower()
            if v in ['stop', 'menu', 'back']: return 'stop'
            if v in ['exit', 'quit']: return 'exit'
            if v == 'status': self.show_status(); return None
            if v == 'history': self.show_history(); return None
            voltage = float(v)
            
            c = input("   Current (A): ").strip().lower()
            if c in ['stop', 'menu', 'back']: return 'stop'
            if c in ['exit', 'quit']: return 'exit'
            current = float(c)
            
            t = input("   Temperature (°C): ").strip().lower()
            if t in ['stop', 'menu', 'back']: return 'stop'
            if t in ['exit', 'quit']: return 'exit'
            temperature = float(t)
            
            if not self._validate(voltage, current, temperature):
                return False
            
            self._add_to_buffer(voltage, current, temperature)
            return True
        except ValueError:
            print("   ❌ Invalid numbers")
            return False
    
    def _validate(self, v, c, t):
        chk = config.PHYSICS_CHECKS
        if not (chk['voltage_absolute_min'] <= v <= chk['voltage_absolute_max']):
            print(f"   ⚠️  Voltage range: {chk['voltage_absolute_min']}-{chk['voltage_absolute_max']}V")
            return False
        if not (chk['current_absolute_min'] <= c <= chk['current_absolute_max']):
            print(f"   ⚠️  Current range: {chk['current_absolute_min']}-{chk['current_absolute_max']}A")
            return False
        if not (chk['temperature_absolute_min'] <= t <= chk['temperature_absolute_max']):
            print(f"   ⚠️  Temperature range: {chk['temperature_absolute_min']}-{chk['temperature_absolute_max']}°C")
            return False
        return True
    
    def _add_to_buffer(self, v, c, t):
        self.buffer['voltage'].append(v)
        self.buffer['current'].append(c)
        self.buffer['temperature'].append(t)
        self.measurement_count += 1
        while len(self.buffer['voltage']) > self.window_size:
            self.buffer['voltage'].pop(0)
            self.buffer['current'].pop(0)
            self.buffer['temperature'].pop(0)
    
    # =========================================================================
    # BUFFER FILL
    # =========================================================================
    
    def _fill_buffer(self):
        """Fill buffer with initial measurements."""
        print("\n" + "="*60)
        print(f"🔄 FILLING BUFFER ({self.window_size} measurements needed)")
        print("="*60)
        
        while len(self.buffer['voltage']) < self.window_size:
            n = len(self.buffer['voltage']) + 1
            r = self.window_size - len(self.buffer['voltage'])
            print(f"\n📝 {n}/{self.window_size} ({r} remaining)")
            
            result = self._input_measurement()
            if result == 'stop': return False
            if result == 'exit': self.running = False; return False
            if result is True:
                print(f"   ✅ {len(self.buffer['voltage'])}/{self.window_size}")
        
        print("\n✅ Buffer full!")
        self.make_prediction()
        return True
    
    # =========================================================================
    # MONITORING
    # =========================================================================
    
    def _monitor(self):
        """Continuous monitoring - fresh buffer each batch."""
        if len(self.buffer['voltage']) < self.window_size:
            if not self._fill_buffer():
                return
        
        print("\n" + "="*60)
        print("🔄 MONITORING")
        print(f"   Each batch collects {self.window_size} new measurements")
        print("   Type 'stop' to exit")
        print("="*60)
        
        batch_num = 1
        
        try:
            while True:
                # Fresh buffer for each batch
                self.buffer = {'voltage': [], 'current': [], 'temperature': []}
                
                print(f"\n{'='*60}")
                print(f"📦 BATCH #{batch_num}")
                print(f"{'='*60}")
                
                for i in range(self.window_size):
                    print(f"\n📝 {i+1}/{self.window_size}")
                    
                    result = self._input_measurement()
                    if result == 'stop': print("\n⏸️  Menu"); return
                    if result == 'exit': return
                    if result is None: continue
                    if result is False: continue
                    
                    print(f"   ✅ {len(self.buffer['voltage'])}/{self.window_size}")
                
                # Predict on complete batch
                print(f"\n⚡ Batch #{batch_num} complete!")
                self.make_prediction()
                batch_num += 1
                
                # Continue?
                print(f"\n{'─'*50}")
                cont = input("   Continue? (Enter=yes, stop=menu, exit=quit): ").strip().lower()
                if cont in ['stop', 'menu', 'back']: print("\n⏸️  Menu"); break
                if cont in ['exit', 'quit']: self.running = False; print("\n🔚 Exit"); return
                
        except KeyboardInterrupt:
            print("\n\n⏸️  Interrupted")
    
    # =========================================================================
    # PREDICTION
    # =========================================================================
    
    def make_prediction(self):
        """Predict fault type from current buffer."""
        if len(self.buffer['voltage']) < self.window_size:
            return
        
        v = np.array(self.buffer['voltage'][-self.window_size:])
        c = np.array(self.buffer['current'][-self.window_size:])
        t = np.array(self.buffer['temperature'][-self.window_size:])
        
        try:
            result = self.predictor.predict_single(v, c, t)
            self.prediction_count += 1
            
            fault = result['fault_type']
            conf = result['confidence']
            symbol, status, _ = self._fault_info(fault)
            
            W = 54
            
            print(f"\n╔{'═'*W}╗")
            print(self._box(f"🔮 PREDICTION #{self.prediction_count}", W))
            print(self._sep(W))
            print(self._box(f"{symbol} {status}", W))
            print(self._box(f"Fault: {fault}", W))
            print(self._box(f"Confidence: {conf:.1%}", W))
            print(self._sep(W))
            print(self._box("TOP-3:", W))
            
            for i, p in enumerate(result['top_3_predictions'][:3]):
                marker = "→" if i == 0 else " "
                prob = p['probability']
                name = p['fault_type'][:18]
                _, _, col = self._fault_info(p['fault_type'])
                
                bar_len = 16
                filled = int(prob * bar_len)
                bar = "█" * filled + "░" * (bar_len - filled)
                
                print(self._box(f"{marker} {name:<18} {col} {bar} {prob:>5.1%}", W))
            
            print(self._sep(W))
            stats = f"V={np.mean(v):.1f}V | I={np.mean(c):.1f}A | T={np.mean(t):.1f}°C"
            print(self._box(stats, W))
            print(f"╚{'═'*W}╝")
            
            if fault != "Normal" and conf >= 0.5:
                print(f"\n🎯 ACTIONS:")
                for i, a in enumerate(self._recommendations(fault), 1):
                    print(f"   {i}. {a}")
            
            self.fault_history.append({
                'pred': self.prediction_count, 'meas': self.measurement_count,
                'fault': fault, 'conf': conf,
                'v': v[-1], 'c': c[-1], 't': t[-1],
            })
            if len(self.fault_history) > 50:
                self.fault_history.pop(0)
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
    
    # =========================================================================
    # STATUS & HISTORY
    # =========================================================================
    
    def show_status(self):
        print(f"\n📊 Buffer: {len(self.buffer['voltage'])}/{self.window_size} | "
              f"Meas: {self.measurement_count} | Pred: {self.prediction_count}")
        if self.buffer['voltage']:
            for i in range(len(self.buffer['voltage'])):
                m = "→" if i == len(self.buffer['voltage'])-1 else " "
                print(f"   [{m}] {self.buffer['voltage'][i]:.1f}V  "
                      f"{self.buffer['current'][i]:.1f}A  "
                      f"{self.buffer['temperature'][i]:.1f}°C")
    
    def show_history(self):
        if not self.fault_history:
            print("\n📊 No history"); return
        
        print(f"\n📊 History ({min(10, len(self.fault_history))}):")
        print("-"*60)
        print(f" {'#':<5} {'Fault':<24} {'Conf':<7} {'V':<6} {'I':<6} {'T':<6}")
        print("-"*60)
        for e in self.fault_history[-10:]:
            s, _, _ = self._fault_info(e['fault'])
            print(f" {e['pred']:<5} {s} {e['fault']:<22} {e['conf']:.1%}  "
                  f"{e['v']:<6.0f} {e['c']:<6.1f} {e['t']:<6.1f}")
        print("-"*60)
    
    def _reset(self):
        self.buffer = {'voltage': [], 'current': [], 'temperature': []}
        self.measurement_count = 0
        self.prediction_count = 0
        self.fault_history = []
        print("🔄 Cleared")
    
    # =========================================================================
    # MAIN
    # =========================================================================
    
    def run(self):
        print("\n" + "="*50)
        print("🔮 FAULT PREDICTION SYSTEM")
        print(f"   Window: {self.window_size} | {config.NOMINAL_VOLTAGE}V | Tanzania")
        print("="*50)
        print("\n  start   - Begin monitoring")
        print("  status  - Buffer status")
        print("  history - Past predictions")
        print("  reset   - Clear all")
        print("  exit    - Quit")
        
        while self.running:
            try:
                c = input("\n> ").strip().lower()
                if c in ['start', 's']:       self._monitor()
                elif c == 'status':           self.show_status()
                elif c == 'history':          self.show_history()
                elif c in ['reset', 'r']:     self._reset()
                elif c in ['exit', 'q']:      self.running = False; print("👋 Bye!")
                elif c == '':                 self._monitor()
                else:                         print(f"❌ '{c}'? Type start/status/history/reset/exit")
            except KeyboardInterrupt:         print("\n⏸️  Interrupted")
            except EOFError:                  self.running = False
        
        return 0


def main():
    try:
        return PredictionMenu().run()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback; traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""
=============================================================================
AGD 307 Radar — Physical Connection Test
=============================================================================
Verified against the AGD 307 Product Manual (Issue 4).

What this script does
─────────────────────
1. Opens the serial port at 9600 baud (factory default, manual p.11).
2. Sends the "AGD" identity command → radar responds with model + firmware.
   If there is NO response, the wiring or switch position is wrong — the
   script will tell you exactly what to check.
3. Queries the current LOWSPEED threshold (*LOWSPEED?) and baud rate (*BAUD?).
4. Sends *MS=5 to enable "dd" format speed streaming at 10 fps (manual p.20).
5. Reads and displays every speed value live in the terminal.

IMPORTANT — why you might see silence after step 4
────────────────────────────────────────────────────
The AGD 307 only emits speed lines when it is actively detecting a vehicle
above the LOWSPEED threshold. Silence between vehicles is NORMAL.
If a vehicle passes and you see nothing, check:
  • Rotary switch must be at position 0 (RS422 mode, manual p.10).
  • DIP switch 1 direction: OFF = advance only, ON = bi-directional.
  • Run with --raw to see every byte the radar sends.

KEY FIX vs earlier version
────────────────────────────
The AGD307 terminates speed lines with \\r (carriage return) ONLY — no \\n.
The buffer normalises all CR / CRLF / LF variants to \\n before splitting,
so speed values are never silently swallowed in the read buffer.

Usage
─────
  python3 test_radar.py                          # /dev/ttyUSB0 @ 9600
  python3 test_radar.py --port /dev/ttyUSB1
  python3 test_radar.py --baud 115200            # if *BAUD=3 was previously set
  python3 test_radar.py --limit 60               # different speed highlight
  python3 test_radar.py --raw                    # also print raw bytes
  python3 test_radar.py --no-init                # skip handshake (already configured)

Press Ctrl+C to exit and see a session summary.
=============================================================================
"""

import re
import sys
import time
import argparse
from datetime import datetime

# ---------------------------------------------------------------------------
# ── ARGUMENT PARSING
# ---------------------------------------------------------------------------
ap = argparse.ArgumentParser(
    description="AGD 307 radar live terminal test",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
ap.add_argument("--port",    default="/dev/ttyUSB0", help="Serial port (default: /dev/ttyUSB0)")
ap.add_argument("--baud",    default=9600, type=int,  help="Baud rate (default: 9600 — AGD 307 factory default)")
ap.add_argument("--limit",   default=40,  type=float, help="Speed limit km/h for highlight (default: 40)")
ap.add_argument("--raw",     action="store_true",     help="Print raw radar lines alongside parsed values")
ap.add_argument("--no-init", action="store_true",     help="Skip AGD handshake and *MS=5 (use if already configured)")
args = ap.parse_args()

# ---------------------------------------------------------------------------
# ── PYSERIAL CHECK
# ---------------------------------------------------------------------------
try:
    import serial
except ImportError:
    print("ERROR: pyserial is not installed.  Run:  pip install pyserial")
    sys.exit(1)

# ---------------------------------------------------------------------------
# ── ANSI COLOURS
# ---------------------------------------------------------------------------
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

# ---------------------------------------------------------------------------
# ── SPEED LINE PARSER
# ── *MS=5 format : bare 2-digit integer  "42"
# ── *MS=6 format : *Sddd                 "*S042"
# ── Legacy       : SPD:dd.d              "SPD:42.3"
# ---------------------------------------------------------------------------
_MS5_RE = re.compile(r"^(\d{2,3})$")
_MS6_RE = re.compile(r"^\*S(\d{3})$")
_SPD_RE = re.compile(r"^SPD:(\d{1,3}(?:\.\d{1,2})?)$")

def parse_speed(line: str) -> float | None:
    """Return speed in km/h or None if the line is not a speed reading."""
    # Skip command echoes and error lines
    if "ERROR" in line.upper():
        return None
    if line.startswith("#") or (line.startswith("*") and not line.startswith("*S")):
        return None
    for pattern in (_MS5_RE, _MS6_RE, _SPD_RE):
        m = pattern.match(line)
        if m:
            try:
                val = float(m.group(1))
                if 0.0 <= val <= 300.0:   # sanity-check: AGD307 max range ~250 km/h
                    return val
            except ValueError:
                pass
    return None

# ---------------------------------------------------------------------------
# ── HEADER
# ---------------------------------------------------------------------------
print()
print("=" * 62)
print("  AGD 307 Radar — Physical Connection Test")
print("=" * 62)
print(f"  Port       : {args.port}")
print(f"  Baud rate  : {args.baud}")
print(f"  Speed limit: {args.limit} km/h  (violations shown in red)")
print(f"  Raw output : {'yes' if args.raw else 'no'}  (add --raw to enable)")
print("=" * 62)
print()

# ---------------------------------------------------------------------------
# ── OPEN PORT
# ---------------------------------------------------------------------------
try:
    ser = serial.Serial(
        port     = args.port,
        baudrate = args.baud,
        bytesize = serial.EIGHTBITS,
        parity   = serial.PARITY_NONE,
        stopbits = serial.STOPBITS_ONE,
        timeout  = 2.0,
        xonxoff  = False,
        rtscts   = False,
        dsrdtr   = False,
    )
except serial.SerialException as exc:
    print(f"{RED}ERROR: Could not open {args.port}{RESET}")
    print(f"       {exc}")
    print()
    print("Things to check:")
    print("  1. Is the FTDI adapter plugged in?")
    print("     ls /dev/ttyUSB*")
    print("  2. Do you have port permission?")
    print("     sudo usermod -aG dialout $USER   (then log out and back in)")
    print("  3. Is the port name correct?")
    print("     dmesg | grep ttyUSB")
    sys.exit(1)

print(f"  {GREEN}Port open.{RESET}\n")

# ---------------------------------------------------------------------------
# ── INITIALISATION SEQUENCE  (manual pp. 16-20)
# ---------------------------------------------------------------------------
def send_cmd(cmd: str, label: str, wait: float = 0.4) -> str:
    """Send a CR-terminated command and return the radar's response."""
    ser.reset_input_buffer()
    ser.write((cmd + "\r").encode("ascii"))
    ser.flush()
    time.sleep(wait)
    raw  = ser.read(ser.in_waiting).decode("ascii", errors="ignore")
    resp = raw.strip().replace("\r\n", " | ").replace("\r", " ").replace("\n", " ")
    print(f"  >> {cmd:<20}  <<  {resp if resp else '(no response)'}")
    return resp

if not args.no_init:
    print("── Step 1/3  AGD identity check ──────────────────────────────────")
    resp = send_cmd("AGD", "identity")
    if not resp:
        print()
        print(f"{RED}  FAILED: Radar did not respond to the AGD command.{RESET}")
        print()
        print("  Checklist:")
        print("  ① Rotary switch inside the radar must be at position 0")
        print("    (RS422 mode — manual p.10). Any other position disables RS422.")
        print("  ② Verify TX+/TX− wiring:")
        print("    Radar Orange (TXZ) → FTDI RX−  (B / orange)")
        print("    Radar Pink   (TXY) → FTDI RX+  (A / yellow)")
        print("  ③ Try other baud rates:  --baud 115200  --baud 19200  --baud 38400")
        print("  ④ Confirm FTDI chip detected:  lsusb | grep FTDI")
        ser.close()
        sys.exit(1)
    print(f"  {GREEN}Radar identified OK.{RESET}\n")

    print("── Step 2/3  Query current settings ──────────────────────────────")
    send_cmd("*BAUD?",     "baud rate query")
    send_cmd("*LOWSPEED?", "low speed threshold query")
    send_cmd("*MS?",       "current message format query")
    send_cmd("*MM?",       "speed measurement mode query")
    print()

    print("── Step 3/3  Enable speed streaming  (*MS=5 → dd @ 10 fps) ───────")
    send_cmd("*MS=5",       "enable speed streaming")
    send_cmd("*LOWSPEED=1", "set low speed threshold to 1 kph")
    print(f"  {GREEN}Speed streaming enabled.{RESET}")
    print()
    print("  NOTE: The radar only sends speed values when it detects a vehicle")
    print("  above the LOWSPEED threshold. Silence between vehicles is normal.")
    print()
else:
    print(f"  {YELLOW}Skipping initialisation (--no-init). Assuming *MS=5 already set.{RESET}\n")

# ---------------------------------------------------------------------------
# ── LIVE READ LOOP
# ──
# ── CRITICAL FIX: The AGD307 terminates speed lines with \r (CR) only —
# ── no \n (LF). The old version split only on \n so all speed data was
# ── silently swallowed in the buffer and never parsed.
# ── Fix: normalise \r\n → \n then \r → \n before splitting.
# ---------------------------------------------------------------------------
ser.timeout = 1.0   # short timeout for responsive reads

print("=" * 62)
print("  Live speed readings  (Ctrl+C to stop)")
print("=" * 62)
print()

total_lines    = 0
valid_readings = 0
violations     = 0
session_max    = 0.0
start_time     = time.monotonic()
last_data_time = time.monotonic()
last_warn_time = 0.0          # throttle the "no data" warning to once per 3 s
buffer         = ""

try:
    while True:
        try:
            chunk = ser.read(ser.in_waiting or 1).decode("ascii", errors="replace")
        except serial.SerialException as exc:
            print(f"\n{RED}Serial error: {exc}{RESET}")
            print("Reconnect the adapter and restart this script.")
            break

        if not chunk:
            # Nothing received — print a gentle reminder (throttled to 3 s)
            now = time.monotonic()
            if valid_readings == 0 and (now - start_time) > 8:
                if now - last_warn_time > 3.0:
                    print(f"  {YELLOW}[No speed data yet — drive a vehicle past the radar]{RESET}")
                    last_warn_time = now
            continue

        last_data_time = time.monotonic()
        buffer += chunk

        # ── KEY FIX ──────────────────────────────────────────────────────
        # AGD307 uses bare \r as line terminator.
        # Normalise \r\n first (so it becomes one \n, not two),
        # then convert any remaining bare \r to \n.
        buffer = buffer.replace("\r\n", "\n").replace("\r", "\n")
        # ─────────────────────────────────────────────────────────────────

        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.strip()
            total_lines += 1

            if not line:
                continue

            speed   = parse_speed(line)
            now_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]

            if speed is not None:
                valid_readings += 1
                session_max = max(session_max, speed)

                if speed > args.limit:
                    violations += 1
                    bar       = "█" * min(int(speed / 2), 40)
                    speed_str = f"{RED}{BOLD}{speed:5.0f} km/h{RESET}"
                    tag       = f"{RED}{BOLD} *** VIOLATION ***{RESET}"
                else:
                    bar       = "▒" * min(int(speed / 2), 40)
                    speed_str = f"{GREEN}{speed:5.0f} km/h{RESET}"
                    tag       = ""

                raw_part = f"  {CYAN}[{line}]{RESET}" if args.raw else ""
                print(f"  {now_str}  {speed_str}  {bar}{tag}{raw_part}")

            else:
                # Line received but not a speed value — command echo or status
                if args.raw:
                    print(f"  {now_str}  {YELLOW}[ctrl]{RESET}  {CYAN}{line}{RESET}")

except KeyboardInterrupt:
    pass
finally:
    ser.close()

# ---------------------------------------------------------------------------
# ── SESSION SUMMARY
# ---------------------------------------------------------------------------
elapsed = time.monotonic() - start_time
print()
print("=" * 62)
print("  Session Summary")
print("=" * 62)
print(f"  Duration         : {elapsed:.1f} s")
print(f"  Lines received   : {total_lines}")
print(f"  Speed readings   : {valid_readings}")
print(f"  Violations       : {violations}  (above {args.limit} km/h)")
print(f"  Session maximum  : {session_max:.0f} km/h")

if total_lines > 0 and valid_readings == 0:
    print()
    print(f"  {YELLOW}WARNING: received {total_lines} lines but parsed 0 speeds.{RESET}")
    print("  The radar is sending data but in an unexpected format.")
    print("  Run with --raw to see exactly what the radar is sending.")
    print("  Then compare it to the *MS command table (manual p.20).")
elif total_lines == 0 and valid_readings == 0:
    print()
    print(f"  {YELLOW}WARNING: no data received at all during the session.{RESET}")
    print("  The radar may not be sending — check wiring and rotary switch.")

print("=" * 62)
print()

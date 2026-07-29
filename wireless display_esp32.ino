/*
  ================================================================
  ESP32 Speed Violation LCD Display
  ================================================================
  Connects to Wi-Fi, runs a TCP server on port 8080.
  Receives plate number + speed from Raspberry Pi.
  Displays on JHD162A 16x2 LCD.

  LCD Row 1:  "KL01 DA 3021    "   (plate number, spaced)
  LCD Row 2:  "80kmph SLOW DOWN"   (speed + warning)

  YOUR WIRING (as tested and working):
    LCD Pin 1  VSS  --> ESP32 GND
    LCD Pin 2  VDD  --> ESP32 5V
    LCD Pin 3  V0   --> Potentiometer middle wiper
    LCD Pin 4  RS   --> ESP32 GPIO23
    LCD Pin 5  RW   --> GND
    LCD Pin 6  EN   --> ESP32 GPIO22
    LCD Pin 11 D4   --> ESP32 GPIO21
    LCD Pin 12 D5   --> ESP32 GPIO19
    LCD Pin 13 D6   --> ESP32 GPIO18
    LCD Pin 14 D7   --> ESP32 GPIO5
    LCD Pin 15 LED+ --> ESP32 5V
    LCD Pin 16 LED- --> ESP32 GND

  Libraries needed:
    - LiquidCrystal  (by Arduino)    -- Sketch > Manage Libraries
    - WiFi           (built-in with ESP32 board package)

  Board settings in Arduino IDE:
    Board         : ESP32 Dev Module
    Upload Speed  : 115200
    CPU Frequency : 240MHz
  ================================================================
*/

#include <WiFi.h>
#include <WiFiServer.h>
#include <LiquidCrystal.h>

// ── Wi-Fi credentials ─────────────────────────────────────────
const char* ssid     = "OPPO F19S";
const char* password = "9446329084";

// ── LCD pin mapping  (RS, EN, D4, D5, D6, D7) ────────────────
LiquidCrystal lcd(23, 22, 21, 19, 18, 5);

// ── TCP server on port 8080 ───────────────────────────────────
WiFiServer server(8080);

// ─────────────────────────────────────────────────────────────
//  formatPlate()
//  Adds spaces to make Indian plates look like:
//  "KL01DA3021" → "KL01 DA 3021"
//  This gives a clean look on the 16-char LCD row.
// ─────────────────────────────────────────────────────────────
String formatPlate(String plate) {
  plate.trim();
  plate.toUpperCase();

  // Standard Indian format: 2 letters + 2 digits + 1-3 letters + 4 digits
  // e.g. KL01DA3021 → KL01 DA 3021
  if (plate.length() >= 8) {
    String part1 = plate.substring(0, 4);          // KL01
    String part3 = plate.substring(plate.length() - 4); // 3021
    String part2 = plate.substring(4, plate.length() - 4); // DA
    return part1 + " " + part2 + " " + part3;      // KL01 DA 3021
  }
  return plate; // return as-is if format unrecognised
}

// ─────────────────────────────────────────────────────────────
//  setup()
// ─────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);

  // Initialise LCD (16 columns, 2 rows)
  lcd.begin(16, 2);
  lcd.clear();

  // ── Show connecting message on LCD ──
  lcd.setCursor(0, 0);
  lcd.print("Connecting WiFi ");
  lcd.setCursor(0, 1);
  lcd.print("Please wait...  ");

  // ── Connect to Wi-Fi ──
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
    attempts++;
    if (attempts > 40) {
      // Timeout after 20 seconds -- show error on LCD
      lcd.clear();
      lcd.setCursor(0, 0);
      lcd.print("WiFi FAILED!    ");
      lcd.setCursor(0, 1);
      lcd.print("Check credentials");
      Serial.println("\nWiFi connection failed! Check SSID and password.");
      while (true) { delay(1000); } // halt
    }
  }

  // ── Connected! Show IP address on LCD ──
  Serial.println("\nWiFi connected!");
  Serial.print("ESP32 IP address: ");
  Serial.println(WiFi.localIP());

  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("WiFi Connected! ");
  lcd.setCursor(0, 1);

  // Print IP on LCD row 2 (note down this IP for the Python script)
  String ip = WiFi.localIP().toString();
  lcd.print(ip);

  // Start the TCP server
  server.begin();

  Serial.println("TCP server started on port 8080.");
  Serial.println(">>> NOTE: Update ESP32_IP in final_test_v4.py with the IP above <<<");

  // Show IP for 4 seconds so you can note it down
  delay(4000);

  // ── Ready state message ──
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("System Ready    ");
  lcd.setCursor(0, 1);
  lcd.print("Waiting data... ");
}

// ─────────────────────────────────────────────────────────────
//  loop()
// ─────────────────────────────────────────────────────────────
void loop() {

  // Check for incoming TCP connection from Raspberry Pi
  WiFiClient client = server.available();

  if (client) {
    Serial.println("Client connected from Raspberry Pi.");
    String received = "";
    long   timeout  = millis();

    // Read until newline '\n' or 2 second timeout
    while ((client.connected() || client.available()) && (millis() - timeout < 2000)) {
      if (client.available()) {
        char c = client.read();
        if (c == '\n') break;   // end of message
        received += c;
        timeout = millis();     // reset timeout on each byte received
      }
    }
    client.stop(); // close connection
    received.trim();

    Serial.print("Received: ");
    Serial.println(received);

    if (received.length() == 0) {
      Serial.println("Empty message received -- ignoring.");
      return;
    }

    // ── Parse message ──
    // Format from Pi: "KL01DA3021|80"
    // Split on '|' to get plate and speed separately
    int sepIndex = received.indexOf('|');

    String plate = "";
    String speed = "";

    if (sepIndex != -1) {
      plate = received.substring(0, sepIndex);
      speed = received.substring(sepIndex + 1);
    } else {
      // No separator -- treat entire message as plate number
      plate = received;
      speed = "";
    }

    plate.trim();
    speed.trim();

    // ── Format plate with spaces for readability ──
    String formattedPlate = formatPlate(plate);

    // ── Build LCD Row 2: "80kmph SLOW DOWN" ──
    String row2 = "";
    if (speed.length() > 0) {
      row2 = speed + "kmph SLOW DOWN";
    } else {
      row2 = "Speed unavailable";
    }

    // ── Truncate both rows to 16 characters (LCD limit) ──
    if (formattedPlate.length() > 16) formattedPlate = formattedPlate.substring(0, 16);
    if (row2.length() > 16)           row2           = row2.substring(0, 16);

    // ── Pad with spaces to clear leftover characters ──
    while (formattedPlate.length() < 16) formattedPlate += " ";
    while (row2.length() < 16)           row2 += " ";

    // ── Write to LCD ──
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print(formattedPlate);   // Row 1: "KL01 DA 3021    "
    lcd.setCursor(0, 1);
    lcd.print(row2);             // Row 2: "80kmph SLOW DOWN"

    Serial.println("LCD updated:");
    Serial.println("  Row1: " + formattedPlate);
    Serial.println("  Row2: " + row2);
  }
}

require('dotenv').config();
'use strict';

const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const cors = require('cors');
const mongoose = require('mongoose');
const mqtt = require('mqtt');
const { v4: uuidv4 } = require('uuid');
const path = require('path');
const fs = require('fs');

const app = express();
const httpServer = http.createServer(app);
const io = new Server(httpServer, {
  cors: { origin: process.env.CORS_ORIGIN || '*' },
});

app.use(cors({ origin: process.env.CORS_ORIGIN || '*' }));
app.use(express.json({ limit: '10mb' }));
app.use(express.static(path.join(__dirname, 'public')));

const PORT = process.env.PORT || 3000;
const MONGODB_URI = process.env.MONGODB_URI || '';
const DEFAULT_SHELF_ID = process.env.DEFAULT_SHELF_ID || 'SHELF_A';
const SEED_DEMO_DATA = String(process.env.SEED_DEMO_DATA || 'false').toLowerCase() === 'true';
const SHELF_W = Number(process.env.SHELF_WIDTH_CM || 12.5);
const SHELF_H = Number(process.env.SHELF_HEIGHT_PER_LEVEL_CM || 12.5);
const SHELF_D = Number(process.env.SHELF_DEPTH_CM || 7.5);
const MQTT_BROKER_URL = process.env.MQTT_BROKER_URL || '';
const MQTT_USERNAME = process.env.MQTT_USERNAME || '';
const MQTT_PASSWORD = process.env.MQTT_PASSWORD || '';
const MQTT_TOPIC_PREFIX = process.env.MQTT_TOPIC_PREFIX || 'warehouse';
const MQTT_LED_RETAIN = String(process.env.MQTT_LED_RETAIN || 'true').toLowerCase() === 'true';
const MQTT_LED_QOS = Number(process.env.MQTT_LED_QOS || 1);
const MQTT_LED_BLINK_MS = Number(process.env.MQTT_LED_BLINK_MS || 500);
const MQTT_LED_TIMEOUT_MS = Number(process.env.MQTT_LED_TIMEOUT_MS || 120000);

let mongoEnabled = false;
let StateModel = null;
let saveTimer = null;
let mqttClient = null;
let mqttReady = false;

// One-document state store. This keeps the current dashboard code simple while
// still persisting all items/events/alerts to MongoDB Atlas when MONGODB_URI is set.
// ── Shelf definitions (source of truth for warehouse layout) ──────────────────
const SHELF_DEFS = [
  { shelf_id: DEFAULT_SHELF_ID, label: 'Kệ A' },
  { shelf_id: 'SHELF_B',        label: 'Kệ B' },
  { shelf_id: 'SHELF_C',        label: 'Kệ C' },
  { shelf_id: 'SHELF_D',        label: 'Kệ D' },
];

// Merge any missing shelves/levels into existing state (e.g. after MongoDB load)
function ensureDefaultShelves(state = db) {
  let changed = false;
  for (const def of SHELF_DEFS) {
    // Add shelf if missing
    if (!state.shelves.find(s => s.shelf_id === def.shelf_id)) {
      state.shelves.push({
        shelf_id: def.shelf_id,
        label: def.label,
        physical_size_cm: { w: SHELF_W, h: SHELF_H * 4, d: SHELF_D },
      });
      console.log(`✚ Added missing shelf: ${def.shelf_id} (${def.label})`);
      changed = true;
    }
    // Add levels T1–T4 for this shelf if missing
    for (let n = 1; n <= 4; n++) {
      if (!state.levels.find(l => l.shelf_id === def.shelf_id && l.level_id === `T${n}`)) {
        state.levels.push({
          level_id: `T${n}`,
          shelf_id: def.shelf_id,
          level_num: n,
          expected_count: 0,
          detected_count: 0,
          capacity_cm: SHELF_W,
          used_cm: 0,
          status: 'ok',
        });
        console.log(`  ✚ Added missing level T${n} for ${def.shelf_id}`);
        changed = true;
      }
    }
  }
  return changed;
}

let db = createInitialState();

function createInitialState() {
  const state = {
    shelves: SHELF_DEFS.map(s => ({
      shelf_id: s.shelf_id,
      label: s.label,
      physical_size_cm: { w: SHELF_W, h: SHELF_H * 4, d: SHELF_D },
    })),
    levels: [],
    items: [],
    events: [],
    alerts: [],
    remove_requests: [],
  };

  for (const sh of SHELF_DEFS) {
    for (let n = 1; n <= 4; n++) {
      state.levels.push({
        level_id: `T${n}`,
        shelf_id: sh.shelf_id,
        level_num: n,
        expected_count: 0,
        detected_count: 0,
        capacity_cm: SHELF_W,
        used_cm: 0,
        status: 'ok',
      });
    }
  }

  if (SEED_DEMO_DATA) seedDemoData(state);
  return state;
}

function seedDemoData(state) {
  const demoItems = [
    { item_id: 'ITEM_DEMO_001', level_id: 'T1', size_cm: { w: 3.2, h: 6.5, d: 3.0 }, x: 0 },
    { item_id: 'ITEM_DEMO_002', level_id: 'T1', size_cm: { w: 4.1, h: 5.8, d: 3.2 }, x: 3.8 },
    { item_id: 'ITEM_DEMO_003', level_id: 'T2', size_cm: { w: 5.0, h: 7.0, d: 3.8 }, x: 0 },
  ];

  for (const it of demoItems) {
    const pos = normalizePosition({ shelf_id: DEFAULT_SHELF_ID, level_id: it.level_id, x_offset_cm: it.x });
    const item = makeItem({
      item_id: it.item_id,
      status: 'placed',
      size_cm: it.size_cm,
      suggested_position: pos,
      placed_position: pos,
    });
    state.items.push(item);
    state.events.unshift({
      event_id: uuidv4(),
      event_type: 'item_placed',
      timestamp: item.created_at,
      shelf_id: DEFAULT_SHELF_ID,
      level_id: it.level_id,
      item_id: item.item_id,
      payload: { size_cm: it.size_cm, position: pos },
    });
  }
  recomputeLevels(state);
}

function makeItem(opts = {}) {
  const id = opts.item_id || `ITEM_${uuidv4().slice(0, 8).toUpperCase()}`;
  const now = opts.created_at ? new Date(opts.created_at) : new Date();
  const size = normalizeSize(opts.size_cm || opts.item_size_cm || { w: 0, h: 0, d: 0 });
  return {
    item_id: id,
    qr_data: opts.qr_data || `WH:${id}`,
    qr_path: opts.qr_path || `/qr/${id}.png`,
    size_cm: size,
    status: opts.status || 'waiting_for_placement',
    suggested_position: normalizePosition(opts.suggested_position),
    placed_position: normalizePosition(opts.placed_position),
    created_at: now.toISOString(),
    updated_at: opts.updated_at || now.toISOString(),
    removed_at: opts.removed_at || null,
  };
}

function normalizeSize(size = {}) {
  return {
    w: Number(size.w ?? size.width ?? size.width_cm ?? 0),
    h: Number(size.h ?? size.height ?? size.height_cm ?? 0),
    d: Number(size.d ?? size.depth ?? size.depth_cm ?? 0),
  };
}

function normalizeShelfId(shelfId) {
  if (!shelfId || shelfId === 'KEA') return DEFAULT_SHELF_ID;
  return String(shelfId);
}

function normalizeLevelId(levelId) {
  if (!levelId) return null;
  const text = String(levelId);
  const match = text.match(/T([1-4])$/i);
  return match ? `T${match[1]}` : text;
}

function levelNumFromId(levelId) {
  const match = String(levelId || '').match(/T([1-4])$/i);
  return match ? Number(match[1]) : null;
}

function normalizePosition(pos) {
  if (!pos) return null;
  const level_id = normalizeLevelId(pos.level_id || pos.level || pos.tier);
  const shelf_id = normalizeShelfId(pos.shelf_id || pos.shelf || DEFAULT_SHELF_ID);
  return {
    ...pos,
    shelf_id,
    level_id,
    level_num: Number(pos.level_num || levelNumFromId(level_id) || 0),
    x_offset_cm: Number(pos.x_offset_cm ?? pos.start_cm ?? pos.x ?? 0),
    start_cm: pos.start_cm !== undefined ? Number(pos.start_cm) : undefined,
    end_cm: pos.end_cm !== undefined ? Number(pos.end_cm) : undefined,
  };
}

// Auto-assign x_offset_cm if it's 0 or missing, so items don't overlap
function autoComputeXOffset(pos, excludeItemId) {
  if (pos && pos.shelf_id && pos.level_id && !pos.x_offset_cm) {
    const lvlItems = db.items.filter(i => 
      i.item_id !== excludeItemId &&
      (i.status === 'placed' || i.status === 'waiting_for_placement')
    );
    let maxX = 0;
    for (const existing of lvlItems) {
      const exPos = existing.placed_position || existing.suggested_position;
      if (exPos && exPos.shelf_id === pos.shelf_id && exPos.level_id === pos.level_id) {
        const exW = existing.size_cm?.w || 3;
        const rightEdge = (exPos.x_offset_cm || 0) + exW;
        if (rightEdge > maxX) maxX = rightEdge;
      }
    }
    if (maxX > 0) pos.x_offset_cm = maxX + 0.5; // 0.5cm gap
  }
  return pos;
}

function ledTopicForShelf(shelfId) {
  return `${MQTT_TOPIC_PREFIX}/${normalizeShelfId(shelfId)}/led/command`;
}

function initMqttBridge() {
  if (!MQTT_BROKER_URL) {
    console.log('MQTT LED bridge disabled. Set MQTT_BROKER_URL to enable HiveMQ publishing.');
    return;
  }

  mqttClient = mqtt.connect(MQTT_BROKER_URL, {
    clientId: process.env.MQTT_CLIENT_ID || `warehouse-cloud-${uuidv4().slice(0, 8)}`,
    clean: true,
    reconnectPeriod: Number(process.env.MQTT_RECONNECT_MS || 5000),
    connectTimeout: Number(process.env.MQTT_CONNECT_TIMEOUT_MS || 30000),
    username: MQTT_USERNAME || undefined,
    password: MQTT_PASSWORD || undefined,
  });

  mqttClient.on('connect', () => {
    mqttReady = true;
    console.log(`MQTT LED bridge connected: ${MQTT_BROKER_URL}`);
  });

  mqttClient.on('offline', () => {
    mqttReady = false;
    console.warn('MQTT LED bridge offline.');
  });

  mqttClient.on('close', () => {
    mqttReady = false;
  });

  mqttClient.on('error', err => {
    mqttReady = false;
    console.error('MQTT LED bridge error:', err.message);
  });
}

function publishLedCommand(command) {
  if (!MQTT_BROKER_URL || !mqttClient) return;
  const topic = ledTopicForShelf(command.shelf_id);
  const payload = JSON.stringify(command);

  mqttClient.publish(
    topic,
    payload,
    { qos: MQTT_LED_QOS, retain: MQTT_LED_RETAIN },
    err => {
      if (err) {
        console.error(`MQTT LED publish failed on ${topic}:`, err.message);
        return;
      }
      console.log(`MQTT LED ${command.command} -> ${topic}: ${payload}`);
    }
  );
}

function publishPlacementBlink(item, evt) {
  const pos = normalizePosition(
    item?.suggested_position ||
    evt?.payload?.suggested_position ||
    evt?.suggested_position
  );
  if (!pos?.level_id) return;

  publishLedCommand({
    command: 'blink',
    shelf_id: pos.shelf_id,
    level_id: pos.level_id,
    item_id: item?.item_id || evt?.item_id || null,
    blink_ms: MQTT_LED_BLINK_MS,
    timeout_ms: MQTT_LED_TIMEOUT_MS,
    source: 'warehouse-cloud',
  });
}

function publishPlacementClear(itemId, position, reason = 'item_placed') {
  const pos = normalizePosition(position || { shelf_id: DEFAULT_SHELF_ID });
  publishLedCommand({
    command: 'clear',
    shelf_id: pos?.shelf_id || DEFAULT_SHELF_ID,
    item_id: itemId || null,
    reason,
    source: 'warehouse-cloud',
  });
}

function ensureShelfAndLevel(shelfId, levelId) {
  const shelf_id = normalizeShelfId(shelfId);
  const level_id = normalizeLevelId(levelId);

  if (!db.shelves.find(s => s.shelf_id === shelf_id)) {
    db.shelves.push({
      shelf_id,
      label: shelf_id,
      physical_size_cm: { w: SHELF_W, h: SHELF_H * 4, d: SHELF_D },
    });
  }

  if (level_id && !db.levels.find(l => l.shelf_id === shelf_id && l.level_id === level_id)) {
    db.levels.push({
      level_id,
      shelf_id,
      level_num: levelNumFromId(level_id) || db.levels.length + 1,
      expected_count: 0,
      detected_count: 0,
      capacity_cm: SHELF_W,
      used_cm: 0,
      status: 'ok',
    });
  }

  return { shelf_id, level_id };
}

function getEventInput(body) {
  const payload = body.payload || {};
  const event_type = body.event_type || body.type;
  const shelf_id = normalizeShelfId(body.shelf_id || payload.shelf_id || payload.shelf);
  const level_id = normalizeLevelId(body.level_id || payload.level_id || payload.level);
  const item_id = body.item_id || payload.item_id || null;
  const timestamp = body.timestamp || payload.timestamp || new Date().toISOString();

  return { event_type, shelf_id, level_id, item_id, timestamp, payload };
}

function extractPosition(input) {
  const p = input.payload || {};
  return normalizePosition(
    p.actual_position ||
    p.placed_position ||
    p.removed_position ||
    p.position ||
    input.actual_position ||
    input.placed_position ||
    input.position ||
    (input.level_id ? { shelf_id: input.shelf_id, level_id: input.level_id } : null)
  );
}

function recomputeLevels(state = db) {
  for (const level of state.levels) {
    const placed = state.items.filter(i => {
      const pos = normalizePosition(i.placed_position);
      return i.status === 'placed' && pos?.shelf_id === level.shelf_id && pos?.level_id === level.level_id;
    });

    level.expected_count = placed.length;
    level.used_cm = Math.round(placed.reduce((sum, item) => sum + Number(item.size_cm?.w || 0), 0) * 10) / 10;

    if (level.detected_count === undefined || level.detected_count === null) {
      level.detected_count = level.expected_count;
    }

    level.status = Number(level.detected_count) === Number(level.expected_count) ? 'ok' : 'warning';
  }
}

function overviewStats() {
  recomputeLevels();
  return {
    total_items: db.items.length,
    placed: db.items.filter(i => i.status === 'placed').length,
    removed: db.items.filter(i => i.status === 'removed').length,
    waiting: db.items.filter(i => i.status === 'waiting_for_placement').length,
    missing_suspected: db.items.filter(i => i.status === 'missing_suspected').length,
    active_warnings: db.alerts.length,
    pending_remove_requests: (db.remove_requests || []).filter(r => r.status === 'pending').length,
    shelves: db.shelves.map(s => shelfStatus(s.shelf_id)),
    storage: mongoEnabled ? 'mongodb_atlas' : 'in_memory',
  };
}

function shelfStatus(shelfId) {
  recomputeLevels();
  const shelf_id = normalizeShelfId(shelfId);
  const shelf = db.shelves.find(s => s.shelf_id === shelf_id);
  if (!shelf) return null;

  const levels = db.levels
    .filter(l => l.shelf_id === shelf_id)
    .sort((a, b) => Number(a.level_num || 0) - Number(b.level_num || 0));

  return {
    ...shelf,
    levels: levels.map(l => ({
      ...l,
      items: db.items.filter(i => {
        const pos = normalizePosition(i.placed_position);
        return i.status === 'placed' && pos?.shelf_id === shelf_id && pos?.level_id === l.level_id;
      }),
    })),
  };
}

function getItemHistory(item_id) {
  return db.events
    .filter(e => e.item_id === item_id)
    .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
}


function normalizeQrLookupText(value) {
  let text = String(value || '').trim();
  try { text = decodeURIComponent(text); } catch (_) {}

  // Accept raw item_id, WH:ITEM_ID, quoted values, or a pasted URL that contains the QR value.
  text = text.replace(/^['"]|['"]$/g, '').trim();
  const urlMatch = text.match(/(?:qr|item|code|id)=([^&\s]+)/i);
  if (urlMatch) {
    text = urlMatch[1];
    try { text = decodeURIComponent(text); } catch (_) {}
  }

  text = text.replace(/^WH:/i, '').trim();
  return text.toUpperCase();
}

function findPlacedItemByQr(qrInput) {
  const key = normalizeQrLookupText(qrInput);
  if (!key) return null;

  return db.items.find(item => {
    const itemId = normalizeQrLookupText(item.item_id);
    const qrData = normalizeQrLookupText(item.qr_data);
    return item.status === 'placed' && (itemId === key || qrData === key);
  }) || null;
}

function getPendingRemoveRequest(itemId) {
  if (!db.remove_requests) db.remove_requests = [];
  return db.remove_requests.find(
    r => r.item_id === itemId && r.status === 'pending'
  ) || null;
}

function createPendingRemoveRequestForItem(item, options = {}) {
  if (!item) {
    const err = new Error('Item not found');
    err.statusCode = 404;
    throw err;
  }

  if (item.status !== 'placed') {
    const err = new Error(`Cannot request removal for item with status: ${item.status}`);
    err.statusCode = 400;
    throw err;
  }

  if (!db.remove_requests) db.remove_requests = [];

  const existingPending = getPendingRemoveRequest(item.item_id);
  if (existingPending) {
    return {
      alreadyPending: true,
      request: existingPending,
      item,
      event: null,
      message: 'Remove request already pending',
    };
  }

  const pos = normalizePosition(item.placed_position || item.suggested_position);
  const now = new Date().toISOString();
  const requestedBy = options.requested_by || 'dashboard';
  const source = options.source || 'dashboard_export';

  const request = {
    request_id: uuidv4(),
    type: 'remove_item',
    status: 'pending',
    item_id: item.item_id,
    qr_data: item.qr_data || `WH:${item.item_id}`,
    shelf_id: pos?.shelf_id || DEFAULT_SHELF_ID,
    level_id: pos?.level_id || null,
    position: pos || null,
    requested_at: now,
    updated_at: now,
    requested_by: requestedBy,
    source,
    note: options.note || '',
  };

  db.remove_requests.unshift(request);

  const evt = {
    event_id: uuidv4(),
    event_type: 'remove_requested',
    timestamp: now,
    shelf_id: request.shelf_id,
    level_id: request.level_id,
    item_id: item.item_id,
    payload: {
      request_id: request.request_id,
      requested_by: requestedBy,
      source,
      note: request.note,
      qr_data: request.qr_data,
      placed_position: item.placed_position,
    },
  };

  pushEvent(evt);
  scheduleSave();

  io.emit('remove_request', request);
  io.emit('outbound_target', {
    item_id: item.item_id,
    request_id: request.request_id,
    shelf_id: request.shelf_id,
    level_id: request.level_id,
    position: request.position,
    source,
  });
  io.emit('overview', overviewStats());

  return {
    alreadyPending: false,
    request,
    item,
    event: evt,
    message: 'Remove request sent to Jetson. Waiting for physical removal confirmation.',
  };
}

function pushEvent(evt) {
  db.events.unshift(evt);
  if (db.events.length > 1000) db.events = db.events.slice(0, 1000);
  io.emit('event', evt);
}

async function persistStateNow() {
  recomputeLevels();
  if (mongoEnabled && StateModel) {
    await StateModel.findOneAndUpdate(
      { key: 'warehouse' },
      { key: 'warehouse', data: db, updated_at: new Date() },
      { upsert: true, new: true }
    );
  } else {
    try {
      fs.writeFileSync('warehouse-db.json', JSON.stringify(db, null, 2), 'utf8');
    } catch (err) {
      console.error('Local JSON save failed:', err.message);
    }
  }
}

function scheduleSave() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    persistStateNow().catch(err => console.error('Save failed:', err.message));
  }, 200);
}

async function initMongo() {
  if (!MONGODB_URI) {
    console.log('ℹ️  MONGODB_URI is not set. Running with local JSON storage.');
    try {
      if (fs.existsSync('warehouse-db.json')) {
        const saved = JSON.parse(fs.readFileSync('warehouse-db.json', 'utf8'));
        if (saved) db = saved;
        ensureDefaultShelves(db);
        console.log('📦 Loaded state from warehouse-db.json');
      }
    } catch (err) {
      console.error('Failed to load local JSON storage:', err.message);
    }
    return;
  }

  const stateSchema = new mongoose.Schema({
    key: { type: String, unique: true, index: true },
    data: { type: mongoose.Schema.Types.Mixed, required: true },
    updated_at: { type: Date, default: Date.now },
  }, { collection: 'warehouse_state', minimize: false });

  StateModel = mongoose.model('WarehouseState', stateSchema);

  await mongoose.connect(MONGODB_URI, {
    serverSelectionTimeoutMS: 10000,
  });

  mongoEnabled = true;
  const saved = await StateModel.findOne({ key: 'warehouse' }).lean();
  if (saved?.data) {
    db = saved.data;
    if (!db.shelves) db.shelves = [];
    if (!db.levels) db.levels = [];
    if (!db.items) db.items = [];
    if (!db.events) db.events = [];
    if (!db.alerts) db.alerts = [];
    if (!db.remove_requests) db.remove_requests = [];
    // Merge any new shelves/levels defined in SHELF_DEFS but missing from DB
    const changed = ensureDefaultShelves(db);
    if (changed) {
      await persistStateNow();
      console.log('✅ Merged new shelves into existing MongoDB state.');
    } else {
      console.log('✅ Loaded warehouse state from MongoDB Atlas.');
    }
  } else {
    await persistStateNow();
    console.log('✅ Created initial warehouse state in MongoDB Atlas.');
  }
}

async function syncFromDB() {
  if (!mongoEnabled || !StateModel) return;
  try {
    const saved = await StateModel.findOne({ key: 'warehouse' }).lean();
    if (saved?.data) {
      db = saved.data;
      ensureDefaultShelves(db);
    }
  } catch (err) {
    console.error('Failed to sync from MongoDB:', err.message);
  }
}

// ─── API Routes ───────────────────────────────────────────────────────────────

app.get('/api/health', (req, res) => {
  res.json({
    ok: true,
    service: 'warehouse-cloud',
    storage: mongoEnabled ? 'mongodb_atlas' : 'in_memory',
    mqtt_led_bridge: MQTT_BROKER_URL ? (mqttReady ? 'connected' : 'configured') : 'disabled',
    time: new Date().toISOString(),
  });
});

// POST /api/events — receive metadata/events from Jetson Nano
app.post('/api/events', async (req, res) => {
  const input = getEventInput(req.body || {});
  if (!input.event_type) return res.status(400).json({ error: 'event_type required' });

  const { shelf_id, level_id } = ensureShelfAndLevel(input.shelf_id, input.level_id);
  const payload = input.payload || {};
  const evt = {
    event_id: req.body.event_id || uuidv4(),
    event_type: input.event_type,
    timestamp: input.timestamp,
    shelf_id: level_id ? shelf_id : (req.body.shelf_id || payload.shelf_id ? shelf_id : null),
    level_id: level_id || null,
    item_id: input.item_id,
    payload,
  };

  if (evt.event_type === 'item_created') {
    const itemId = evt.item_id || `ITEM_${uuidv4().slice(0, 8).toUpperCase()}`;
    let item = db.items.find(i => i.item_id === itemId);
    if (!item) {
      item = makeItem({
        item_id: itemId,
        status: 'waiting_for_placement',
        size_cm: payload.size_cm || payload.item_size_cm || req.body.size_cm,
        suggested_position: payload.suggested_position || req.body.suggested_position,
        created_at: evt.timestamp,
      });
      db.items.unshift(item);
    } else {
      item.status = 'waiting_for_placement';
      item.size_cm = normalizeSize(payload.size_cm || payload.item_size_cm || item.size_cm);
      item.suggested_position = normalizePosition(payload.suggested_position || item.suggested_position);
      item.updated_at = evt.timestamp;
    }
    evt.item_id = itemId;

    const suggested = normalizePosition(payload.suggested_position || req.body.suggested_position || item.suggested_position);
    if (suggested?.level_id) {
      ensureShelfAndLevel(suggested.shelf_id, suggested.level_id);
      autoComputeXOffset(suggested, itemId);
      item.suggested_position = suggested;
      evt.shelf_id = suggested.shelf_id;
      evt.level_id = suggested.level_id;
      if (!evt.payload.suggested_position) evt.payload.suggested_position = suggested;
    }

    pushEvent(evt);
    publishPlacementBlink(item, evt);
    scheduleSave();
    io.emit('overview', overviewStats());
    return res.status(201).json({ event: evt, item, storage: mongoEnabled ? 'mongodb_atlas' : 'in_memory' });
  }

  if (evt.event_type === 'item_placed') {
    const itemId = evt.item_id || `ITEM_${uuidv4().slice(0, 8).toUpperCase()}`;
    const position = autoComputeXOffset(extractPosition({ ...input, shelf_id, level_id }), itemId);
    let item = db.items.find(i => i.item_id === itemId);
    if (!item) {
      item = makeItem({
        item_id: itemId,
        status: 'placed',
        size_cm: payload.size_cm || payload.item_size_cm || req.body.size_cm,
        suggested_position: payload.suggested_position || position,
        placed_position: position,
        created_at: evt.timestamp,
      });
      db.items.unshift(item);
    } else {
      item.status = 'placed';
      item.size_cm = normalizeSize(payload.size_cm || payload.item_size_cm || item.size_cm);
      item.placed_position = position || autoComputeXOffset(normalizePosition({ shelf_id, level_id }), itemId);
      item.suggested_position = normalizePosition(payload.suggested_position || item.suggested_position);
      item.updated_at = evt.timestamp;
      item.removed_at = null;
    }
    evt.item_id = itemId;
    if (!evt.payload.position && position) evt.payload.position = position;
    publishPlacementClear(itemId, position || item.suggested_position, 'item_placed');
  }

  if (evt.event_type === 'item_removed') {
  const item = db.items.find(i => i.item_id === evt.item_id);
  if (item) {
    item.status = 'removed';
    item.updated_at = evt.timestamp;
    item.removed_at = evt.timestamp;
  }

  if (!db.remove_requests) db.remove_requests = [];
  for (const reqItem of db.remove_requests) {
    if (reqItem.item_id === evt.item_id && reqItem.status === 'pending') {
      reqItem.status = 'completed';
      reqItem.updated_at = evt.timestamp;
      reqItem.completed_at = evt.timestamp;
      reqItem.completed_by = 'jetson';
    }
  }
}

  if (evt.event_type === 'inventory_count_warning') {
    const lvl = db.levels.find(l => l.shelf_id === shelf_id && l.level_id === level_id);
    if (lvl) {
      lvl.detected_count = Number(payload.detected_count ?? lvl.detected_count ?? 0);
      lvl.expected_count = Number(payload.expected_count ?? lvl.expected_count ?? 0);
      lvl.status = 'warning';
    }
    db.alerts.unshift({ ...evt });
    if (db.alerts.length > 100) db.alerts = db.alerts.slice(0, 100);
  }

  if (evt.event_type === 'inventory_status') {
    const updates = Array.isArray(payload.levels) ? payload.levels : [{ shelf_id, level_id, ...payload }];
    for (const update of updates) {
      const ns = normalizeShelfId(update.shelf_id || shelf_id);
      const nl = normalizeLevelId(update.level_id || level_id);
      ensureShelfAndLevel(ns, nl);
      const lvl = db.levels.find(l => l.shelf_id === ns && l.level_id === nl);
      if (!lvl) continue;
      lvl.detected_count = Number(update.detected_count ?? lvl.detected_count ?? 0);
      lvl.expected_count = Number(update.expected_count ?? lvl.expected_count ?? lvl.expected_count);
      lvl.status = lvl.detected_count === lvl.expected_count ? 'ok' : 'warning';
      if (lvl.status === 'ok') {
        db.alerts = db.alerts.filter(a => !(a.shelf_id === ns && a.level_id === nl));
      }
    }
  }

  recomputeLevels();

  // For normal placed/removed transactions, Jetson has confirmed the visual
  // state, so expected_count and detected_count should match until a later
  // inventory_count_warning/inventory_status says otherwise.
  if (evt.event_type === 'item_placed' || evt.event_type === 'item_removed') {
    const eventLevelId = evt.level_id || normalizePosition(db.items.find(i => i.item_id === evt.item_id)?.placed_position)?.level_id;
    const eventShelfId = evt.shelf_id || normalizePosition(db.items.find(i => i.item_id === evt.item_id)?.placed_position)?.shelf_id || DEFAULT_SHELF_ID;
    const lvl = db.levels.find(l => l.shelf_id === eventShelfId && l.level_id === eventLevelId);
    if (lvl) {
      lvl.detected_count = lvl.expected_count;
      lvl.status = 'ok';
      db.alerts = db.alerts.filter(a => !(a.shelf_id === eventShelfId && a.level_id === eventLevelId));
    }
  }

  pushEvent(evt);
  scheduleSave();
  io.emit('overview', overviewStats());
  res.status(201).json({ event: evt, storage: mongoEnabled ? 'mongodb_atlas' : 'in_memory' });
});

app.get('/api/items', async (req, res) => {
  await syncFromDB();
  const { status, shelf_id } = req.query;
  let items = [...db.items];
  if (status) items = items.filter(i => i.status === status);
  if (shelf_id) {
    const sid = normalizeShelfId(shelf_id);
    items = items.filter(i => normalizePosition(i.placed_position)?.shelf_id === sid || normalizePosition(i.suggested_position)?.shelf_id === sid);
  }
  res.json({ items, total: items.length });
});

app.get('/api/items/:item_id', (req, res) => {
  const item = db.items.find(i => i.item_id === req.params.item_id);
  if (!item) return res.status(404).json({ error: 'Item not found' });
  res.json({ item, events: getItemHistory(item.item_id) });
});

app.post('/api/items/:item_id/remove-request', async (req, res) => {
  const item = db.items.find(i => i.item_id === req.params.item_id);

  try {
    const result = createPendingRemoveRequestForItem(item, {
      requested_by: 'dashboard',
      source: 'dashboard_export',
      note: req.body?.note || '',
    });

    res.status(result.alreadyPending ? 200 : 202).json({
      success: true,
      message: result.message,
      request: result.request,
      item: result.item,
      event: result.event,
    });
  } catch (err) {
    res.status(err.statusCode || 500).json({ error: err.message || 'Remove request failed' });
  }
});

app.post('/api/qr/lookup', async (req, res) => {
  await syncFromDB();
  const qrInput = req.body?.qr_data || req.body?.qr || req.body?.code || req.body?.item_id || '';
  const normalized = normalizeQrLookupText(qrInput);

  if (!normalized) {
    return res.status(400).json({ error: 'QR data is required' });
  }

  const anyItem = db.items.find(item => {
    const itemId = normalizeQrLookupText(item.item_id);
    const qrData = normalizeQrLookupText(item.qr_data);
    return itemId === normalized || qrData === normalized;
  });

  if (!anyItem) {
    return res.status(404).json({ error: 'No item found for this QR code', normalized });
  }

  if (anyItem.status !== 'placed') {
    return res.status(400).json({
      error: `Item is not available for outbound because status is ${anyItem.status}`,
      item: anyItem,
      normalized,
    });
  }

  res.json({
    success: true,
    normalized,
    item: anyItem,
    position: normalizePosition(anyItem.placed_position || anyItem.suggested_position),
    pending_request: getPendingRemoveRequest(anyItem.item_id),
  });
});

app.post('/api/qr/outbound-request', async (req, res) => {
  await syncFromDB();
  const qrInput = req.body?.qr_data || req.body?.qr || req.body?.code || req.body?.item_id || '';
  const normalized = normalizeQrLookupText(qrInput);

  if (!normalized) {
    return res.status(400).json({ error: 'QR data is required' });
  }

  const item = findPlacedItemByQr(qrInput);
  if (!item) {
    return res.status(404).json({
      error: 'No placed item found for this QR code',
      normalized,
    });
  }

  try {
    const result = createPendingRemoveRequestForItem(item, {
      requested_by: 'dashboard_qr_search',
      source: 'qr_lookup',
      note: req.body?.note || 'QR search outbound request',
    });

    res.status(result.alreadyPending ? 200 : 202).json({
      success: true,
      message: result.message,
      normalized,
      item: result.item,
      position: normalizePosition(result.item.placed_position || result.item.suggested_position),
      request: result.request,
      event: result.event,
    });
  } catch (err) {
    res.status(err.statusCode || 500).json({ error: err.message || 'QR outbound request failed' });
  }
});

app.get('/api/remove-requests', (req, res) => {
  if (!db.remove_requests) db.remove_requests = [];

  const status = req.query.status || null;
  let requests = [...db.remove_requests];

  if (status) {
    requests = requests.filter(r => r.status === status);
  }

  res.json({
    requests,
    total: requests.length,
  });
});

app.get('/api/shelves/:shelf_id/status', async (req, res) => {
  await syncFromDB();
  const status = shelfStatus(req.params.shelf_id);
  if (!status) return res.status(404).json({ error: 'Shelf not found' });
  res.json(status);
});

app.get('/api/alerts', async (req, res) => {
  await syncFromDB();
  res.json({ alerts: db.alerts, total: db.alerts.length });
});

app.get('/api/overview', async (req, res) => {
  await syncFromDB();
  res.json(overviewStats());
});

app.get('/api/events', async (req, res) => {
  await syncFromDB();
  const limit = Number.parseInt(req.query.limit, 10) || 50;
  res.json({ events: db.events.slice(0, limit), total: db.events.length });
});

// Keep SPA routing working when deployed publicly.
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// ─── Socket.IO ────────────────────────────────────────────────────────────────
io.on('connection', socket => {
  socket.emit('overview', overviewStats());
  socket.emit('connected', {
    message: 'Connected to Warehouse Cloud',
    storage: mongoEnabled ? 'mongodb_atlas' : 'in_memory',
    time: new Date().toISOString(),
  });
});

setInterval(() => {
  io.emit('jetson_heartbeat', {
    timestamp: new Date().toISOString(),
    status: 'online',
    edge_device: 'Jetson Nano',
  });
}, 15000);

async function start() {
  try {
    await initMongo();
  } catch (err) {
    console.error('❌ MongoDB Atlas connection failed:', err.message);
    console.error('   Check MONGODB_URI, Database User credentials, and Network Access/IP whitelist.');
    process.exit(1);
  }

  initMqttBridge();

  httpServer.listen(PORT, '0.0.0.0', () => {
    console.log(`\n🚀 Warehouse Cloud running on port ${PORT}`);
    console.log(`   Local:  http://localhost:${PORT}`);
    console.log(`   Public: use your hosting provider URL`);
    console.log(`   Storage: ${mongoEnabled ? 'MongoDB Atlas' : 'Local JSON'}\n`);
  });
}

process.on('SIGTERM', async () => {
  try { await persistStateNow(); } catch (_) {}
  try { if (mqttClient) mqttClient.end(true); } catch (_) {}
  process.exit(0);
});

start();

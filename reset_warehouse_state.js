require('dotenv').config();
const mongoose = require('mongoose');

const DEFAULT_SHELF_ID = process.env.DEFAULT_SHELF_ID || 'SHELF_A';

function createEmptyState() {
  const state = {
    shelves: [
      {
        shelf_id: DEFAULT_SHELF_ID,
        label: 'Kệ A',
        physical_size_cm: { w: 12.5, h: 50, d: 7.5 },
      },
    ],
    levels: [],
    items: [],
    events: [],
    alerts: [],
  };

  for (let n = 1; n <= 4; n++) {
    state.levels.push({
      level_id: `T${n}`,
      shelf_id: DEFAULT_SHELF_ID,
      level_num: n,
      expected_count: 0,
      detected_count: 0,
      capacity_cm: 12.5,
      used_cm: 0,
      status: 'ok',
    });
  }

  return state;
}

async function main() {
  if (!process.env.MONGODB_URI) {
    throw new Error('MONGODB_URI is not set');
  }

  await mongoose.connect(process.env.MONGODB_URI);

  console.log('Connected DB:', mongoose.connection.db.databaseName);

  const emptyState = createEmptyState();

  await mongoose.connection.db.collection('warehouse_state').updateOne(
    { key: 'warehouse' },
    {
      $set: {
        key: 'warehouse',
        data: emptyState,
        updated_at: new Date(),
      },
    },
    { upsert: true }
  );

  console.log('✅ Warehouse state reset successfully.');
  console.log('Items: 0');
  console.log('Events: 0');
  console.log('Alerts: 0');

  await mongoose.disconnect();
}

main().catch(async (err) => {
  console.error('❌ Reset failed:', err.message);
  try { await mongoose.disconnect(); } catch (_) {}
  process.exit(1);
});

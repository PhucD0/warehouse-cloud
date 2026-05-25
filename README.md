# WarehouseOS — Smart Warehouse Cloud Backend

Cloud backend + web dashboard cho hệ thống Smart Warehouse IoT với Jetson Nano edge device.

Jetson Nano xử lý computer vision ở edge. Cloud chỉ nhận metadata/event, lưu vào MongoDB Atlas và phát realtime lên web dashboard bằng Socket.IO.

---

## 1. Cấu trúc hiện tại

```text
warehouse-cloud/
├── server.js              # Express API + Socket.IO + MongoDB Atlas persistence
├── public/index.html      # Dashboard SPA
├── package.json
├── package-lock.json
├── Procfile
└── .env.example
```

Project này là một Node.js/Express app duy nhất: backend API và frontend dashboard chạy chung một domain. Khi host public, dashboard tự gọi API theo `window.location.origin`, nên không cần hard-code localhost.

---

## 2. Chạy local

```bash
npm install
npm start
```

Mặc định nếu chưa có `MONGODB_URI`, server vẫn chạy bằng in-memory storage:

```text
http://localhost:3000
```

Kiểm tra health:

```bash
curl http://localhost:3000/api/health
```

---

## 3. MongoDB Atlas

Tạo file `.env` từ `.env.example` hoặc cấu hình environment variables trên hosting provider:

```env
MONGODB_URI=mongodb+srv://<db_user>:<db_password>@<cluster-host>/warehouse?retryWrites=true&w=majority
DEFAULT_SHELF_ID=SHELF_A
SEED_DEMO_DATA=false
CORS_ORIGIN=*
```

Lưu ý:

- Database user phải có quyền đọc/ghi database `warehouse`.
- Network Access trong Atlas phải cho phép IP của hosting provider kết nối tới cluster.
- Nếu hosting provider không có IP cố định, có thể tạm dùng `0.0.0.0/0` cho demo, nhưng không nên dùng lâu dài cho production.

Dữ liệu được lưu trong collection:

```text
warehouse.warehouse_state
```

Backend hiện lưu toàn bộ dashboard state trong một document có `key = "warehouse"`. Cách này đơn giản, phù hợp demo realtime. Sau này có thể tách riêng thành collections `items`, `events`, `alerts`, `shelves` nếu cần scale lớn hơn.

---

## 4. Deploy public trên Render

1. Đưa project này lên GitHub.
2. Vào Render Dashboard.
3. New → Web Service → Connect GitHub repo.
4. Runtime: Node.
5. Build Command:

```bash
npm install
```

6. Start Command:

```bash
npm start
```

7. Add Environment Variables:

```env
MONGODB_URI=mongodb+srv://<db_user>:<db_password>@<cluster-host>/warehouse?retryWrites=true&w=majority
DEFAULT_SHELF_ID=SHELF_A
SEED_DEMO_DATA=false
CORS_ORIGIN=*
```

8. Deploy và lấy public URL, ví dụ:

```text
https://warehouse-cloud-demo.onrender.com
```

Sau khi deploy, test:

```bash
curl https://YOUR_PUBLIC_URL/api/health
```

---

## 5. Deploy public trên Railway

1. Đưa project này lên GitHub.
2. Railway → New Project → Deploy from GitHub repo.
3. Chọn repo chứa project.
4. Thêm Variables:

```env
MONGODB_URI=mongodb+srv://<db_user>:<db_password>@<cluster-host>/warehouse?retryWrites=true&w=majority
DEFAULT_SHELF_ID=SHELF_A
SEED_DEMO_DATA=false
CORS_ORIGIN=*
```

5. Deploy và lấy public domain trong Settings/Domains.

---

## 6. API chính

### Health check

```http
GET /api/health
```

### Dashboard overview

```http
GET /api/overview
```

### Danh sách item

```http
GET /api/items
GET /api/items?status=placed
```

### Chi tiết item

```http
GET /api/items/:item_id
```

### Trạng thái kệ

```http
GET /api/shelves/SHELF_A/status
```

### Cảnh báo

```http
GET /api/alerts
```

### Event log

```http
GET /api/events?limit=100
```

### Jetson gửi event

```http
POST /api/events
Content-Type: application/json
```

---

## 7. Schema event từ Jetson

### item_created

```json
{
  "event_type": "item_created",
  "timestamp": "2026-05-24T10:30:00Z",
  "item_id": "ITEM_000001",
  "payload": {
    "size_cm": { "w": 4.5, "h": 8.2, "d": 3.1 },
    "suggested_position": {
      "shelf_id": "SHELF_A",
      "level_id": "T1",
      "x_offset_cm": 0
    }
  }
}
```

### item_placed

```json
{
  "event_type": "item_placed",
  "timestamp": "2026-05-24T10:31:00Z",
  "item_id": "ITEM_000001",
  "shelf_id": "SHELF_A",
  "level_id": "T1",
  "payload": {
    "size_cm": { "w": 4.5, "h": 8.2, "d": 3.1 },
    "actual_position": {
      "shelf_id": "SHELF_A",
      "level_id": "T1",
      "x_offset_cm": 0,
      "start_cm": 0,
      "end_cm": 4.5
    }
  }
}
```

### inventory_count_warning

```json
{
  "event_type": "inventory_count_warning",
  "timestamp": "2026-05-24T10:32:00Z",
  "shelf_id": "SHELF_A",
  "level_id": "T1",
  "payload": {
    "expected_count": 2,
    "detected_count": 1,
    "status": "suspected_missing_or_merged",
    "message": "expected_count and detected_count mismatch"
  }
}
```

### item_removed

```json
{
  "event_type": "item_removed",
  "timestamp": "2026-05-24T10:35:00Z",
  "item_id": "ITEM_000001",
  "shelf_id": "SHELF_A",
  "level_id": "T1",
  "payload": {
    "removed_position": {
      "shelf_id": "SHELF_A",
      "level_id": "T1"
    }
  }
}
```

### inventory_status

Một tầng:

```json
{
  "event_type": "inventory_status",
  "shelf_id": "SHELF_A",
  "level_id": "T1",
  "payload": {
    "expected_count": 2,
    "detected_count": 2
  }
}
```

Nhiều tầng:

```json
{
  "event_type": "inventory_status",
  "payload": {
    "levels": [
      { "shelf_id": "SHELF_A", "level_id": "T1", "expected_count": 2, "detected_count": 2 },
      { "shelf_id": "SHELF_A", "level_id": "T2", "expected_count": 1, "detected_count": 1 }
    ]
  }
}
```

---

## 8. Test nhanh sau khi deploy

Thay `YOUR_PUBLIC_URL` bằng domain thật:

```bash
curl -X POST https://YOUR_PUBLIC_URL/api/events \
  -H "Content-Type: application/json" \
  -d '{
    "event_type":"item_created",
    "item_id":"ITEM_TEST_001",
    "payload":{
      "size_cm":{"w":4.5,"h":8.2,"d":3.1},
      "suggested_position":{"shelf_id":"SHELF_A","level_id":"T1","x_offset_cm":0}
    }
  }'
```

```bash
curl -X POST https://YOUR_PUBLIC_URL/api/events \
  -H "Content-Type: application/json" \
  -d '{
    "event_type":"item_placed",
    "item_id":"ITEM_TEST_001",
    "shelf_id":"SHELF_A",
    "level_id":"T1",
    "payload":{
      "size_cm":{"w":4.5,"h":8.2,"d":3.1},
      "actual_position":{"shelf_id":"SHELF_A","level_id":"T1","x_offset_cm":0,"start_cm":0,"end_cm":4.5}
    }
  }'
```

Dashboard phải cập nhật realtime qua Socket.IO.

---

## 9. Jetson Nano integration

Trên Jetson, đặt:

```python
CLOUD_URL = "https://YOUR_PUBLIC_URL"
```

Mỗi lần `append_event()` ghi local vào `warehouse_events.jsonl`, gọi thêm:

```python
requests.post(f"{CLOUD_URL}/api/events", json=event, timeout=5)
```

Chỉ gửi metadata/event. Không gửi video thô.

import streamlit as st
import pandas as pd
import pdfplumber
import io
import re
import math
from datetime import datetime, timedelta

st.set_page_config(page_title="Deposit Manager Ultimate AI", layout="wide")
st.title("🏨 ระบบจัดการ Deposit (Luxury Escape)")

DEFAULT_POOL_COLS = ['Select', 'Booking ID', 'Guest Name', 'Amount', 'Date', 'Checkin_Date'] 
DEFAULT_OPERA_COLS = ['Select', 'Voucher', 'Guest Name', 'Amount', 'Date', 'Checkin_Date']
HISTORY_COLS = ['Action_Date', 'Type', 'Booking_ID/Voucher', 'Guest Name', 'Amount', 'Ref_Date', 'Source']

if 'deposit_pool' not in st.session_state:
    st.session_state.deposit_pool = pd.DataFrame(columns=DEFAULT_POOL_COLS)
if 'unpaid_opera' not in st.session_state:
    st.session_state.unpaid_opera = pd.DataFrame(columns=DEFAULT_OPERA_COLS)
if 'match_history' not in st.session_state:
    st.session_state.match_history = pd.DataFrame(columns=HISTORY_COLS)
if 'history_stack' not in st.session_state:
    st.session_state.history_stack = []

if 'cum_dep' not in st.session_state: st.session_state.cum_dep = 0
if 'cum_op' not in st.session_state: st.session_state.cum_op = 0

def save_state():
    state = {
        'deposit_pool': st.session_state.deposit_pool.copy(),
        'unpaid_opera': st.session_state.unpaid_opera.copy(),
        'match_history': st.session_state.match_history.copy()
    }
    st.session_state.history_stack.append(state)
    if len(st.session_state.history_stack) > 10:
        st.session_state.history_stack.pop(0)

def restore_state():
    if st.session_state.history_stack:
        state = st.session_state.history_stack.pop()
        st.session_state.deposit_pool = state['deposit_pool']
        st.session_state.unpaid_opera = state['unpaid_opera']
        st.session_state.match_history = state['match_history']
        st.toast("↩️ ย้อนกลับเรียบร้อย")
        st.rerun()
    else:
        st.warning("ไม่มีรายการให้ย้อนกลับ")

def parse_date_manual(date_str, format_type='DMY'):
    try:
        if pd.isna(date_str) or str(date_str).strip() == '': return None
        s = str(date_str).strip().split(' ')[0]
        if '/' in s:
            parts = s.split('/')
            if len(parts) == 3:
                p1, p2, p3 = int(parts[0]), int(parts[1]), int(parts[2])
                if p3 < 100: p3 += 2000
                if format_type == 'DMY': return datetime(p3, p2, p1).date()
                else:
                    if p1 > 1000: return datetime(p1, p2, p3).date()
                    return datetime(p3, p1, p2).date()
        if '-' in s: return datetime.strptime(s, "%Y-%m-%d").date()
    except: pass
    try: return pd.to_datetime(s, dayfirst=(format_type=='DMY')).date()
    except: return None

def clean_money(val):
    if pd.isna(val) or str(val).strip() == '': return 0.0
    s = str(val).strip().replace(",", "")
    try: return float(s)
    except: return 0.0

def clean_id(val):
    text = str(val).strip().upper()
    if text in ['NAN', 'NONE', '']: return ''
    if text.endswith('.0'): text = text[:-2]
    if '.' in text: text = text.split('.')[0]
    return text

def ensure_state_structure():
    for col in DEFAULT_POOL_COLS:
        if col not in st.session_state.deposit_pool.columns: st.session_state.deposit_pool[col] = None
    for col in DEFAULT_OPERA_COLS:
        if col not in st.session_state.unpaid_opera.columns: st.session_state.unpaid_opera[col] = None
    st.session_state.deposit_pool['Select'] = st.session_state.deposit_pool['Select'].fillna(False).astype(bool)
    st.session_state.unpaid_opera['Select'] = st.session_state.unpaid_opera['Select'].fillna(False).astype(bool)

def log_history(df, action_type, source):
    if df.empty: return
    records = []
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    for _, row in df.iterrows():
        ref_id = row['Booking ID'] if source == 'Deposit' else row['Voucher']
        records.append({
            'Action_Date': now_str, 'Type': action_type, 
            'Booking_ID/Voucher': ref_id, 'Guest Name': row.get('Guest Name', ''),
            'Amount': row['Amount'], 'Ref_Date': row['Date'], 'Source': source
        })
    new_hist = pd.DataFrame(records)
    st.session_state.match_history = pd.concat([st.session_state.match_history, new_hist], ignore_index=True)

def find_recommendations(pool, unpaid):
    # --- 1. Preparation: Force numeric types for calculation ---
    pool = pool.copy()
    unpaid = unpaid.copy()
    pool['Amount'] = pd.to_numeric(pool['Amount'], errors='coerce').fillna(0.0)
    unpaid['Amount'] = pd.to_numeric(unpaid['Amount'], errors='coerce').fillna(0.0)

    recs = []
    pool_map = {} 
    # Map Booking ID -> List of indices
    for idx, row in pool.iterrows():
        key = clean_id(row['Booking ID'])
        if key: pool_map.setdefault(key, []).append(idx)

    covered_op = set(); covered_dep = set()

    # --- 2. ID Match Logic (Scanning) ---
    for o_idx, o_row in unpaid.iterrows():
        v = clean_id(o_row['Voucher'])
        if v and v in pool_map:
            # Gather all candidates for this Voucher ID
            candidates = []
            for p_idx in pool_map[v]:
                if p_idx in covered_dep: continue
                p_row = pool.loc[p_idx]
                diff = abs(p_row['Amount'] - o_row['Amount'])
                candidates.append((diff, p_idx, p_row))
            
            if candidates:
                # Sort candidates by difference (smallest difference first)
                candidates.sort(key=lambda x: x[0])
                
                # Pick the best candidate
                best_diff, best_p_idx, best_p_row = candidates[0]
                
                # Create recommendation
                is_perfect = (best_diff <= 5.0) # Relaxed threshold slightly for float diffs
                reason = "✅ เลขตรง + ยอดตรง (Perfect)" if is_perfect else f"⚠️ เลขตรงแต่ยอดไม่เท่า (Diff {best_p_row['Amount'] - o_row['Amount']:,.2f})"
                
                recs.append({
                    'Select': False,
                    'Opera_Idx': o_idx, 'Dep_Idx': best_p_idx,
                    'Op_Voucher': o_row['Voucher'],      
                    'Dep_Booking_ID': best_p_row['Booking ID'], 
                    'Op_Amount': o_row['Amount'], 'Dep_Amount': best_p_row['Amount'],
                    'Reason': reason, 
                    'Op_Guest': o_row['Guest Name'], 'Dep_Guest': best_p_row['Guest Name'],
                    'Op_Date': o_row['Checkin_Date'], 'Dep_Date': best_p_row['Checkin_Date']
                })
                
                # CRITICAL CHANGE: Only mark as covered if the match is "Good Enough".
                # If ID matches but amount is wildly different, we show it as a warning (rec),
                # BUT we do NOT mark it as covered. This allows other logics (Exact Amount/Name)
                # to pick up the item if there's a better match available.
                if is_perfect:
                    covered_op.add(o_idx)
                    covered_dep.add(best_p_idx)

    # --- 3. Exact Amount Match (Strict 1-to-1) ---
    # Only for items not covered by ID match
    
    # FIX: Use .groups to get Index Labels instead of .indices (which gives integer positions)
    # This prevents "off-by-one" or wrong row selection when rows are filtered/deleted.
    pool_subset = pool[~pool.index.isin(covered_dep) & (pool['Amount'] > 0)]
    op_subset = unpaid[~unpaid.index.isin(covered_op) & (unpaid['Amount'] > 0)]

    pool_groups = pool_subset.groupby('Amount').groups
    op_groups = op_subset.groupby('Amount').groups
    
    for amt, pool_idxs in pool_groups.items():
        if amt in op_groups:
            op_idxs = op_groups[amt]
            if len(pool_idxs) == 1 and len(op_idxs) == 1:
                p_idx = pool_idxs[0]; o_idx = op_idxs[0]
                recs.append({
                    'Select': False, 'Opera_Idx': o_idx, 'Dep_Idx': p_idx,
                    'Op_Amount': amt, 'Dep_Amount': amt,
                    'Reason': '💰 ยอดเงินตรงกันเป๊ะ (No ID)',
                    'Op_Voucher': unpaid.at[o_idx, 'Voucher'], 'Dep_Booking_ID': pool.at[p_idx, 'Booking ID'],
                    'Op_Guest': unpaid.at[o_idx, 'Guest Name'], 'Dep_Guest': pool.at[p_idx, 'Guest Name'],
                    'Op_Date': unpaid.at[o_idx, 'Checkin_Date'], 'Dep_Date': pool.at[p_idx, 'Checkin_Date']
                })
                covered_op.add(o_idx); covered_dep.add(p_idx)
    
    # --- 4. Name Similarity Match ---
    for o_idx, o_row in unpaid.iterrows():
        if o_idx in covered_op: continue
        o_name = str(o_row['Guest Name']).upper()
        o_tokens = set(re.split(r'\s+', re.sub(r'[^A-Z]', ' ', o_name))) - {'MR', 'MRS', 'MS', 'MISS'}
        if not o_tokens: continue
        
        for p_idx, p_row in pool.iterrows():
            if p_idx in covered_dep: continue
            
            # Skip if amount difference is too large (e.g. > 100) to avoid false positives on common names
            if abs(p_row['Amount'] - o_row['Amount']) > 100: continue 
            
            p_name = str(p_row['Guest Name']).upper()
            p_tokens = set(re.split(r'\s+', re.sub(r'[^A-Z]', ' ', p_name))) - {'MR', 'MRS', 'MS', 'MISS'}
            
            if len(o_tokens.intersection(p_tokens)) >= 1:
                date_match = False
                if o_row['Checkin_Date'] and p_row['Checkin_Date']:
                    try:
                        # Ensure dates are comparable
                        if abs((o_row['Checkin_Date'] - p_row['Checkin_Date']).days) <= 3: date_match = True
                    except: pass
                
                reason = ""
                if date_match: reason = "📅 ชื่อคล้าย + วันใกล้กัน"
                elif abs(p_row['Amount'] - o_row['Amount']) < 1.0: reason = "💰 ชื่อคล้าย + ยอดตรง"
                
                if reason:
                    recs.append({
                        'Select': False, 'Opera_Idx': o_idx, 'Dep_Idx': p_idx,
                        'Op_Voucher': o_row['Voucher'], 
                        'Dep_Booking_ID': p_row['Booking ID'],
                        'Op_Amount': o_row['Amount'], 'Dep_Amount': p_row['Amount'], 'Reason': reason,
                        'Op_Guest': o_row['Guest Name'], 'Dep_Guest': p_row['Guest Name'],
                        'Op_Date': o_row['Checkin_Date'], 'Dep_Date': p_row['Checkin_Date']
                    })
                    covered_op.add(o_idx); covered_dep.add(p_idx)
                    break 
    return pd.DataFrame(recs)

TABLE_HEADER_REGEX = re.compile(r"\bDate\s+Folio\s+Description\s+Arrival\s+Departure\s+Voucher\s+Debit\s+Credit\s+Balance\b", re.IGNORECASE)
FOOTER_BREAKERS = [r"^Balance\b", r"^Due\b", r"^Aging Summary", r"^Bank Details", r"^STATEMENT OF ACCOUNT", r"^TRIP\.COM", r"^Page\s+\d+\s+of\s+\d+", r"^A/R Account", r"^Print Date", r"^Tel:|Kata Thani|The Shore|TAX ID"]

def _page_lines(page):
    full = page.extract_text(x_tolerance=1, y_tolerance=2) or ""; lines = [ln.rstrip() for ln in full.split("\n")]; header_idx = -1
    for i, ln in enumerate(lines):
        if TABLE_HEADER_REGEX.search(ln): header_idx = i; break
    if header_idx == -1: return []
    out = []; started = False
    for i, txt in enumerate(lines):
        txt = txt.strip(); 
        if i == header_idx: started = True; continue
        if not started: continue
        if any(re.match(p, txt, re.IGNORECASE) for p in FOOTER_BREAKERS): break
        out.append(txt)
    return out

def _parse_opera_row(line):
    m = re.match(r"^(?P<date>\d{2}/\d{2}/\d{2})\s+(?P<folio>\d+)\s+(?P<rest>.+)$", line)
    if not m: return None
    raw_date = m.group("date"); rest = m.group("rest")
    monies = re.findall(r"[\d,]+\.\d{2}", rest)
    if not monies: return None
    amount = 0.0
    try:
        if len(monies) >= 1: amount = clean_money(monies[0])
        if len(monies) >= 2: amount = clean_money(monies[-2])
    except: pass
    if amount <= 0: return None
    dates_in_rest = re.findall(r"\d{2}/\d{2}/\d{2}", rest)
    voucher = ""; guest = ""; checkin_date = None
    if len(dates_in_rest) >= 2:
        arr_str = dates_in_rest[0]; dep_str = dates_in_rest[1]
        checkin_date = parse_date_manual(arr_str, 'DMY')
        idx_arr = rest.find(arr_str); guest = rest[:idx_arr].strip()
        idx_dep = rest.find(dep_str); after_dep = rest[idx_dep + len(dep_str):].strip()
        tokens = after_dep.split()
        possible_vouchers = []
        for t in tokens:
            if re.match(r"[\d,]+\.\d{2}", t): break
            possible_vouchers.append(t)
        if possible_vouchers: voucher = possible_vouchers[-1]
    return {"Select": False, "Date": parse_date_manual(raw_date, 'DMY'), "Checkin_Date": checkin_date, "Voucher": clean_id(voucher), "Guest Name": guest, "Amount": amount}

def load_opera_pdf(file):
    rows = []
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                lines = _page_lines(page)
                for ln in lines:
                    r = _parse_opera_row(ln)
                    if r and r['Date']: rows.append(r)
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=DEFAULT_OPERA_COLS)
    except Exception as e: st.error(f"Error PDF: {e}"); return pd.DataFrame(columns=DEFAULT_OPERA_COLS)

def load_deposit_file(file):
    try:
        if file.name.lower().endswith('.csv'):
            try: df = pd.read_csv(file, encoding='utf-8-sig', header=None)
            except: file.seek(0); df = pd.read_csv(file, encoding='cp1252', header=None)
        else: df = pd.read_excel(file, header=None)
        header_idx = -1
        for i, row in df.head(50).iterrows():
            if 'BOOKING ID' in row.astype(str).str.cat(sep=' ').upper() and ('AMOUNT' in row.astype(str).str.cat(sep=' ').upper() or 'ADMIN' in row.astype(str).str.cat(sep=' ').upper()): header_idx = i; break
        if header_idx == -1: return pd.DataFrame(columns=DEFAULT_POOL_COLS)
        if file.name.lower().endswith('.csv'): file.seek(0); df = pd.read_csv(file, skiprows=header_idx, encoding='utf-8-sig')
        else: df = pd.read_excel(file, skiprows=header_idx)
        cols = [str(c).strip() for c in df.columns]; df.columns = cols
        
        def find_col(k):
            for c in cols: 
                if any(x.upper() in c.upper() for x in k): return c
            return None
            
        c_id = find_col(['Booking ID']); c_paid = find_col(['Amount paid']); c_credit = find_col(['Amount to credit']); c_fee = find_col(['Admin Fee']); c_guest = find_col(['Primary guest name', 'Guest Name', 'Guest'])
        c_pay_date = None
        for c in cols:
            if 'PAYMENT' in c.upper() and 'DATE' in c.upper(): c_pay_date = c; break
        c_chk_date = find_col(['Check-in Date', 'Checkin', 'Arrival'])
        
        if not c_id: return pd.DataFrame(columns=DEFAULT_POOL_COLS)
        data = pd.DataFrame()
        data['Booking ID'] = df[c_id].astype(str).str.strip()
        data['Guest Name'] = df[c_guest].astype(str).str.strip() if c_guest else ""
        v_paid = df[c_paid].apply(clean_money).fillna(0) if c_paid else 0.0
        v_cred = df[c_credit].apply(clean_money).fillna(0) if c_credit else 0.0
        v_fee  = df[c_fee].apply(clean_money).fillna(0) if c_fee else 0.0
        data['Amount'] = v_paid + v_cred + v_fee
        data['Date'] = df[c_pay_date].apply(lambda x: parse_date_manual(x, 'MDY')) if c_pay_date else None
        data['Checkin_Date'] = df[c_chk_date].apply(lambda x: parse_date_manual(x, 'MDY')) if c_chk_date else None
        data['Select'] = False; data = data[data['Booking ID'].str.len() > 1]
        
        final = data.groupby(['Booking ID', 'Date'], as_index=False).agg({
            'Guest Name': 'first', 
            'Amount': 'sum', 
            'Checkin_Date': 'first', 
            'Select': 'first'
        })
        return final
    except Exception as e: st.error(f"Error reading file {file.name}: {e}"); return pd.DataFrame(columns=DEFAULT_POOL_COLS)

with st.sidebar:
    st.header("📂 จัดการไฟล์")
    if st.button("↩️ ย้อนกลับ (Undo)", type="secondary"):
        restore_state()
    st.write("---")
    if st.button("🗑️ ล้างข้อมูลใหม่ (Reset)", type="primary"):
        save_state()
        st.session_state.deposit_pool = pd.DataFrame(columns=DEFAULT_POOL_COLS)
        st.session_state.unpaid_opera = pd.DataFrame(columns=DEFAULT_OPERA_COLS)
        st.session_state.match_history = pd.DataFrame(columns=HISTORY_COLS)
        st.session_state.history_stack = []
        st.session_state.cum_dep = 0; st.session_state.cum_op = 0; st.rerun()
    st.write("---")
    old_file = st.file_uploader("1. ยอดยกมา (Data_Store)", type=['xlsx'])
    if old_file and st.button("โหลด Master File"):
        try:
            x = pd.ExcelFile(old_file); dp = pd.read_excel(x, 'Deposit_Pool'); op = pd.read_excel(x, 'Unpaid_Opera')
            if 'Match_History' in x.sheet_names: st.session_state.match_history = pd.read_excel(x, 'Match_History')
            for df in [dp, op]: 
                if 'Date' in df.columns: df['Date'] = pd.to_datetime(df['Date']).dt.date
                if 'Checkin_Date' in df.columns: df['Checkin_Date'] = pd.to_datetime(df['Checkin_Date']).dt.date
            st.session_state.deposit_pool = dp; st.session_state.unpaid_opera = op; ensure_state_structure()
            st.session_state.cum_dep = len(dp); st.session_state.cum_op = len(op)
            st.success(f"✅ โหลดเรียบร้อย: Deposit {len(dp)} / Opera {len(op)}")
        except Exception as e: st.error(f"Error: {e}")
    deposits = st.file_uploader("2. ไฟล์ Deposit (Excel/CSV)", accept_multiple_files=True)
    if deposits and st.button(f"โหลด Deposit ({len(deposits)})"):
        dfs = []; 
        for f in deposits: dfs.append(load_deposit_file(f))
        new_data = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        if not new_data.empty:
            save_state()
            st.session_state.deposit_pool = pd.concat([st.session_state.deposit_pool, new_data], ignore_index=True)
            st.session_state.deposit_pool = st.session_state.deposit_pool.groupby(['Booking ID', 'Date'], as_index=False).agg({'Guest Name': 'first', 'Amount': 'sum', 'Checkin_Date': 'first', 'Select': 'first'})
            st.session_state.cum_dep += len(new_data)
            pos_amt = new_data[new_data['Amount'] > 0]['Amount'].sum()
            neg_amt = new_data[new_data['Amount'] < 0]['Amount'].sum()
            st.success(f"✅ เพิ่ม {len(new_data)} รายการ\n- ยอดรับ (Income): {pos_amt:,.2f}\n- ยอดหัก (Refund): {neg_amt:,.2f}")
    operas = st.file_uploader("3. ไฟล์ Opera (PDF)", type=['pdf'], accept_multiple_files=True)
    if operas and st.button(f"โหลด Opera ({len(operas)})"):
        dfs = []; 
        for f in operas: dfs.append(load_opera_pdf(f))
        new_data = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        if not new_data.empty:
            save_state()
            st.session_state.unpaid_opera = pd.concat([st.session_state.unpaid_opera, new_data], ignore_index=True)
            st.session_state.cum_op += len(new_data)
            added_amt = new_data['Amount'].sum()
            st.success(f"✅ เพิ่ม Opera {len(new_data)} รายการ (รวม {added_amt:,.2f})")
            
    st.write("---")
    with st.expander("➕ เพิ่มรายการพิเศษ"):
        with st.form("manual"):
            b_id = st.text_input("Ref ID"); amt = st.number_input("Amount"); g_name = st.text_input("Name"); d_date = st.date_input("Date")
            side = st.radio("Side:", ["Deposit", "Opera"]); submit = st.form_submit_button("บันทึก")
            if submit:
                save_state()
                row = {'Select': False, 'Booking ID' if side=="Deposit" else 'Voucher': b_id, 'Guest Name': g_name, 'Amount': amt, 'Date': d_date, 'Checkin_Date': d_date}
                if side=="Deposit": st.session_state.deposit_pool = pd.concat([st.session_state.deposit_pool, pd.DataFrame([row])], ignore_index=True); st.session_state.cum_dep += 1
                else: st.session_state.unpaid_opera = pd.concat([st.session_state.unpaid_opera, pd.DataFrame([row])], ignore_index=True); st.session_state.cum_op += 1
                ensure_state_structure(); st.rerun()

ensure_state_structure()
m1, m2, m3 = st.columns(3)
m1.metric("💰 Deposit คงเหลือ", f"{st.session_state.deposit_pool['Amount'].sum():,.2f}")
m2.metric("🔴 Opera คงค้าง", f"{st.session_state.unpaid_opera['Amount'].sum():,.2f}")
m3.metric("⚖️ Net Balance", f"{(st.session_state.deposit_pool['Amount'].sum() - st.session_state.unpaid_opera['Amount'].sum()):,.2f}")
st.markdown("---")

t1, t2, t3, t4, t5 = st.tabs(["⚡ Auto Match", "✋ Manual Match", "📜 History", "📊 Dashboard", "💾 Save"])

with t1: 
    col_stat1, col_stat2 = st.columns(2)
    with col_stat1: st.info(f"🔴 **Opera**\n\nโหลด: {st.session_state.cum_op} | เหลือ: {len(st.session_state.unpaid_opera)}")
    with col_stat2: st.success(f"🟢 **Deposit**\n\nโหลด: {st.session_state.cum_dep} | เหลือ: {len(st.session_state.deposit_pool)}")
    if st.button("🚀 เริ่มจับคู่ (Auto - Strict Exact Match)", type="primary"):
        save_state()
        pool = st.session_state.deposit_pool.copy(); unpaid = st.session_state.unpaid_opera.copy()
        pool['Amount'] = pd.to_numeric(pool['Amount'], errors='coerce').fillna(0.0)
        unpaid['Amount'] = pd.to_numeric(unpaid['Amount'], errors='coerce').fillna(0.0)
        pool_map = {}
        for idx, row in pool.iterrows(): pool_map.setdefault(clean_id(row['Booking ID']), []).append(idx)
        drop_pool = []; drop_unpaid = []; matched_pool = []; matched_op = []
        for i, row in unpaid.iterrows():
            v = clean_id(row['Voucher']); match_idx = None
            if v in pool_map:
                for p in pool_map[v]:
                    if p not in drop_pool and abs(pool.at[p, 'Amount'] - row['Amount']) < 1.0: 
                        match_idx = p; break
            if match_idx is not None:
                drop_unpaid.append(i); drop_pool.append(match_idx)
                matched_op.append(row); matched_pool.append(pool.loc[match_idx])
        if matched_pool: log_history(pd.DataFrame(matched_pool), 'Auto', 'Deposit'); log_history(pd.DataFrame(matched_op), 'Auto', 'Opera')
        st.session_state.deposit_pool = pool.drop(drop_pool).reset_index(drop=True)
        st.session_state.unpaid_opera = unpaid.drop(drop_unpaid).reset_index(drop=True)
        st.success(f"จับคู่สำเร็จ {len(drop_unpaid)} รายการ (ยอดตรงเป๊ะ)"); st.rerun()

with t2: 
    with st.expander("🔍 ตัวช่วยค้นหา & กรองข้อมูล (Smart Filter)", expanded=False):
        c_search1, c_search2, c_search3 = st.columns(3)
        search_amt = c_search1.number_input("ค้นหายอดเงิน (Amount)", value=0.0, step=100.0)
        search_name = c_search2.text_input("ค้นหาชื่อแขก (Name)")
        search_reset = c_search3.button("ล้างตัวกรอง")

    view_op = st.session_state.unpaid_opera.copy()
    view_dep = st.session_state.deposit_pool.copy()
    
    if search_amt > 0:
        view_op = view_op[ (view_op['Amount'] >= search_amt-1) & (view_op['Amount'] <= search_amt+1) ]
        view_dep = view_dep[ (view_dep['Amount'] >= search_amt-1) & (view_dep['Amount'] <= search_amt+1) ]
    if search_name:
        view_op = view_op[view_op['Guest Name'].astype(str).str.contains(search_name, case=False, na=False)]
        view_dep = view_dep[view_dep['Guest Name'].astype(str).str.contains(search_name, case=False, na=False)]

    if st.button("🪄 สแกนหาคู่ที่น่าสงสัย (Magic Scan)", help="ค้นหารายการที่เลขตรงแต่ยอดไม่เท่า หรือ ชื่อคล้าย"):
        recs = find_recommendations(st.session_state.deposit_pool, st.session_state.unpaid_opera)
        if not recs.empty: st.session_state['recommendations'] = recs; st.toast(f"เจอ {len(recs)} คู่!")
        else: st.info("ไม่พบคู่ที่น่าสงสัย")

    if 'recommendations' in st.session_state and not st.session_state['recommendations'].empty:
        st.markdown("### 💡 รายการแนะนำ (เลือกคู่ที่ต้องการ)")
        edited_recs = st.data_editor(
            st.session_state['recommendations'],
            column_order=["Select", "Reason", "Op_Voucher", "Dep_Booking_ID", "Op_Amount", "Dep_Amount", "Op_Guest", "Dep_Guest", "Op_Date", "Dep_Date"],
            column_config={
                "Select": st.column_config.CheckboxColumn("เลือก", default=False),
                "Op_Amount": st.column_config.NumberColumn("ยอด Opera", format="%.2f"),
                "Dep_Amount": st.column_config.NumberColumn("ยอด Deposit", format="%.2f"),
                "Reason": st.column_config.TextColumn("สาเหตุ", width="medium"),
            },
            use_container_width=True, hide_index=True, key="recs_editor"
        )
        c1, c2 = st.columns([1, 5])
        if c1.button("🔗 จับคู่ที่เลือก"):
            sel = edited_recs[edited_recs['Select']]
            if not sel.empty:
                save_state()
                op_idx_list = sel['Opera_Idx'].tolist(); dep_idx_list = sel['Dep_Idx'].tolist()
                
                valid_op = [i for i in op_idx_list if i in st.session_state.unpaid_opera.index]
                valid_dep = [i for i in dep_idx_list if i in st.session_state.deposit_pool.index]
                
                if len(valid_op) != len(op_idx_list) or len(valid_dep) != len(dep_idx_list):
                    st.error("บางรายการถูกลบไปแล้ว กรุณากดสแกนใหม่")
                else:
                    drop_op = []
                    drop_dep = []
                    
                    processed_op = set()
                    processed_dep = set()
                    
                    for i in range(len(sel)):
                        row_val = sel.iloc[i]
                        o_idx = op_idx_list[i]
                        d_idx = dep_idx_list[i]
                        
                        # Prevent processing duplicates if user selects multiple rows for same item
                        if o_idx in processed_op or d_idx in processed_dep:
                            continue
                        
                        processed_op.add(o_idx)
                        processed_dep.add(d_idx)
                        
                        deduct_amt = row_val['Op_Amount'] 
                        
                        log_history(st.session_state.unpaid_opera.loc[[o_idx]], 'Magic Match', 'Opera')
                        
                        current_dep_amt = st.session_state.deposit_pool.at[d_idx, 'Amount']
                        new_dep_amt = current_dep_amt - deduct_amt
                        st.session_state.deposit_pool.at[d_idx, 'Amount'] = new_dep_amt
                        
                        drop_op.append(o_idx)
                        if abs(new_dep_amt) < 0.01: 
                            drop_dep.append(d_idx)
                            
                    st.session_state.unpaid_opera = st.session_state.unpaid_opera.drop(drop_op).reset_index(drop=True)
                    st.session_state.deposit_pool = st.session_state.deposit_pool.drop(drop_dep).reset_index(drop=True)
                    st.session_state['recommendations'] = pd.DataFrame() 
                    st.success("จับคู่และตัดยอดเรียบร้อย!"); st.rerun()
        if c2.button("❌ ปิด"): del st.session_state['recommendations']; st.rerun()
        st.markdown("---")

    s_op = st.session_state.unpaid_opera[st.session_state.unpaid_opera['Select']]['Amount'].sum()
    s_dep = st.session_state.deposit_pool[st.session_state.deposit_pool['Select']]['Amount'].sum()
    
    c_info1, c_info2 = st.columns(2)
    c_info1.info(f"🔴 Opera เลือก: **{s_op:,.2f}**")
    c_info2.success(f"🟢 Deposit เลือก: **{s_dep:,.2f}**")

    # --- NO FORM HERE FOR REAL-TIME EDIT ---
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown(f"##### 🔴 Opera ({len(view_op)})")
        edited_op = st.data_editor(
            view_op,
            column_order=["Select", "Voucher", "Guest Name", "Amount", "Checkin_Date"],
            column_config={"Select": st.column_config.CheckboxColumn(width="small"), "Checkin_Date": st.column_config.DateColumn("Arrival", format="DD/MM/YYYY"), "Amount": st.column_config.NumberColumn(format="%.2f")},
            hide_index=True, key="editor_op", height=400
        )
    with col_r:
        st.markdown(f"##### 🟢 Deposit ({len(view_dep)})")
        edited_dep = st.data_editor(
            view_dep,
            column_order=["Select", "Booking ID", "Guest Name", "Amount", "Checkin_Date"],
            column_config={"Select": st.column_config.CheckboxColumn(width="small"), "Checkin_Date": st.column_config.DateColumn("Check-in", format="DD/MM/YYYY"), "Amount": st.column_config.NumberColumn(format="%.2f")},
            hide_index=True, key="editor_dep", height=400
        )
    
    # --- AUTO-SYNC LOGIC ---
    # Compare edited df vs session state df to detect changes
    has_changes = False
    
    # Check for Opera changes
    for idx in view_op.index:
        if idx in edited_op.index:
            new_amt = clean_money(edited_op.loc[idx, 'Amount'])
            if st.session_state.unpaid_opera.at[idx, 'Amount'] != new_amt:
                st.session_state.unpaid_opera.at[idx, 'Amount'] = new_amt
                has_changes = True
            
            # Sync other fields if needed
            st.session_state.unpaid_opera.at[idx, 'Guest Name'] = edited_op.loc[idx, 'Guest Name']
            st.session_state.unpaid_opera.at[idx, 'Voucher'] = edited_op.loc[idx, 'Voucher']
            
            # Sync Selection
            if edited_op.loc[idx, 'Select'] != st.session_state.unpaid_opera.at[idx, 'Select']:
                 st.session_state.unpaid_opera.at[idx, 'Select'] = edited_op.loc[idx, 'Select']
                 has_changes = True

    # Check for Deposit changes
    for idx in view_dep.index:
        if idx in edited_dep.index:
            new_amt = clean_money(edited_dep.loc[idx, 'Amount'])
            if st.session_state.deposit_pool.at[idx, 'Amount'] != new_amt:
                st.session_state.deposit_pool.at[idx, 'Amount'] = new_amt
                has_changes = True

            st.session_state.deposit_pool.at[idx, 'Guest Name'] = edited_dep.loc[idx, 'Guest Name']
            st.session_state.deposit_pool.at[idx, 'Booking ID'] = edited_dep.loc[idx, 'Booking ID']
            
            # Sync Selection
            if edited_dep.loc[idx, 'Select'] != st.session_state.deposit_pool.at[idx, 'Select']:
                 st.session_state.deposit_pool.at[idx, 'Select'] = edited_dep.loc[idx, 'Select']
                 has_changes = True

    if has_changes:
        st.rerun()

    c_btn1, c_btn2, c_btn3 = st.columns([1, 1, 1])
    match_click = c_btn2.button("🔗 จับคู่ที่เลือก (ตัดยอด)", type="primary")
    delete_click = c_btn3.button("🗑️ ลบทิ้ง")

    if match_click:
        sel_op_idx = st.session_state.unpaid_opera[st.session_state.unpaid_opera['Select']].index
        sel_dep_idx = st.session_state.deposit_pool[st.session_state.deposit_pool['Select']].index
        
        if len(sel_op_idx) > 0 and len(sel_dep_idx) > 0:
            save_state()
            if len(sel_dep_idx) == 1:
                dep_target_idx = sel_dep_idx[0]
                total_op_amount = st.session_state.unpaid_opera.loc[sel_op_idx, 'Amount'].sum()
                current_dep_val = st.session_state.deposit_pool.at[dep_target_idx, 'Amount']
                new_dep_val = current_dep_val - total_op_amount
                
                st.session_state.deposit_pool.at[dep_target_idx, 'Amount'] = new_dep_val
                
                log_history(st.session_state.unpaid_opera.loc[sel_op_idx], 'Manual', 'Opera')
                log_history(st.session_state.deposit_pool.loc[sel_dep_idx], 'Manual', 'Deposit')
                
                st.session_state.unpaid_opera = st.session_state.unpaid_opera.drop(sel_op_idx).reset_index(drop=True)
                
                if abs(new_dep_val) < 0.01:
                    st.session_state.deposit_pool = st.session_state.deposit_pool.drop(sel_dep_idx).reset_index(drop=True)
                else:
                    st.session_state.deposit_pool.loc[dep_target_idx, 'Select'] = False
                
                st.toast("Success! Deposit updated.")
                st.rerun()
            elif len(sel_op_idx) == len(sel_dep_idx):
                drop_op = []
                drop_dep = []
                op_list = sel_op_idx.tolist()
                dep_list = sel_dep_idx.tolist()
                
                for i in range(len(op_list)):
                    oid = op_list[i]
                    did = dep_list[i]
                    op_val = st.session_state.unpaid_opera.at[oid, 'Amount']
                    dep_val = st.session_state.deposit_pool.at[did, 'Amount']
                    new_val = dep_val - op_val
                    
                    st.session_state.deposit_pool.at[did, 'Amount'] = new_val
                    drop_op.append(oid)
                    
                    if abs(new_val) < 0.01:
                        drop_dep.append(did)
                        
                log_history(st.session_state.unpaid_opera.loc[drop_op], 'Manual', 'Opera')
                log_history(st.session_state.deposit_pool.loc[dep_list], 'Manual', 'Deposit')
                
                st.session_state.unpaid_opera = st.session_state.unpaid_opera.drop(drop_op).reset_index(drop=True)
                if drop_dep:
                    st.session_state.deposit_pool = st.session_state.deposit_pool.drop(drop_dep).reset_index(drop=True)
                
                st.session_state.deposit_pool['Select'] = False
                st.toast("Success! Pairs updated.")
                st.rerun()
            else:
                st.warning("เลือก Deposit 1 รายการเพื่อตัดยอด หรือเลือกจับคู่จำนวนเท่ากัน")
        else:
            st.warning("เลือกรายการทั้ง 2 ฝั่ง")

    if delete_click:
        sel_op_idx = st.session_state.unpaid_opera[st.session_state.unpaid_opera['Select']].index
        sel_dep_idx = st.session_state.deposit_pool[st.session_state.deposit_pool['Select']].index
        
        if len(sel_op_idx) > 0 or len(sel_dep_idx) > 0:
            save_state()
            if len(sel_op_idx) > 0:
                log_history(st.session_state.unpaid_opera.loc[sel_op_idx], 'Deleted', 'Opera')
                st.session_state.unpaid_opera = st.session_state.unpaid_opera.drop(sel_op_idx).reset_index(drop=True)
            if len(sel_dep_idx) > 0:
                log_history(st.session_state.deposit_pool.loc[sel_dep_idx], 'Deleted', 'Deposit')
                st.session_state.deposit_pool = st.session_state.deposit_pool.drop(sel_dep_idx).reset_index(drop=True)
            st.toast("Deleted!"); st.rerun()
        else:
            st.warning("เลือกรายการที่จะลบก่อน")

    c_all_op, c_un_op, c_all_dp, c_un_dp = st.columns(4)
    if c_all_op.button("Select All (Op)"):
        st.session_state.unpaid_opera.loc[view_op.index, 'Select'] = True; st.rerun()
    if c_un_op.button("Unselect (Op)"):
        st.session_state.unpaid_opera['Select'] = False; st.rerun()
    if c_all_dp.button("Select All (Dep)"):
        st.session_state.deposit_pool.loc[view_dep.index, 'Select'] = True; st.rerun()
    if c_un_dp.button("Unselect (Dep)"):
        st.session_state.deposit_pool['Select'] = False; st.rerun()

with t3: 
    if not st.session_state.match_history.empty:
        df_show = st.session_state.match_history.copy()
        df_show['Ref_Date'] = pd.to_datetime(df_show['Ref_Date'], errors='coerce')
        st.dataframe(df_show, use_container_width=True)
    else: st.info("Empty")

with t4: 
    curr = st.session_state.deposit_pool.copy(); hist = st.session_state.match_history.copy()
    curr['DT'] = pd.to_datetime(curr['Date'], errors='coerce'); hist['DT'] = pd.to_datetime(hist['Ref_Date'], errors='coerce')
    curr['M'] = curr['DT'].dt.to_period('M').astype(str); hist['M'] = hist['DT'].dt.to_period('M').astype(str)
    
    h_dep = hist[hist['Source']=='Deposit']; total = pd.concat([curr[['M','Amount']], h_dep[['M','Amount']]])
    g_in = total.groupby('M')['Amount'].sum().reset_index().rename(columns={'Amount':'Inflow'})
    g_used = hist[(hist['Source']=='Deposit') & (hist['Type'].str.contains('Match'))].groupby('M')['Amount'].sum().reset_index().rename(columns={'Amount':'Used'})
    g_rem = curr.groupby('M')['Amount'].sum().reset_index().rename(columns={'Amount':'Remain'})
    
    fin = g_in.merge(g_used, on='M', how='outer').merge(g_rem, on='M', how='outer').fillna(0).sort_values('M')
    st.bar_chart(fin.set_index('M')); st.dataframe(fin, use_container_width=True)

with t5: 
    b = io.BytesIO()
    with pd.ExcelWriter(b, engine='xlsxwriter') as w:
        st.session_state.deposit_pool.drop(columns=['Select']).to_excel(w, sheet_name='Deposit_Pool', index=False)
        st.session_state.unpaid_opera.drop(columns=['Select']).to_excel(w, sheet_name='Unpaid_Opera', index=False)
        st.session_state.match_history.to_excel(w, sheet_name='Match_History', index=False)
    st.download_button("📥 Download Excel", b, "Data_Store.xlsx", "application/vnd.ms-excel", type="primary")

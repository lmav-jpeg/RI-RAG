CREATE TABLE IF NOT EXISTS connections (
    id INT AUTO_INCREMENT UNIQUE,
    uid VARCHAR(255) NOT NULL,
    file_origin VARCHAR(255) NOT NULL,
    stream VARCHAR(50) NULL,
    day INT NOT NULL,

    -- Optional fields (default to NULL if missing)
    duration DOUBLE NULL,
    orig_bytes BIGINT NULL,
    resp_bytes BIGINT NULL,
    service VARCHAR(100) NULL,

    -- Additional Zeek log fields
    ts DOUBLE NULL,
    system_name VARCHAR(255) NULL,
    proc VARCHAR(50) NULL,
    id_orig_h VARCHAR(45) NULL,
    id_orig_p INT NULL,
    id_resp_h VARCHAR(45) NULL,
    id_resp_p INT NULL,
    proto VARCHAR(10) NULL,
    conn_state VARCHAR(10) NULL,
    missed_bytes BIGINT NULL,
    history VARCHAR(100) NULL,
    orig_pkts BIGINT NULL,
    orig_ip_bytes BIGINT NULL,
    resp_pkts BIGINT NULL,
    resp_ip_bytes BIGINT NULL,
    orig_l2_addr VARCHAR(17) NULL,
    resp_l2_addr VARCHAR(17) NULL,

    -- Composite Primary Key as requested
    PRIMARY KEY (uid, file_origin)
) ENGINE=InnoDB;
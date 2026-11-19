# TechNova Inc. — IT Policies & Security Guidelines

## Password Policy

All employees must follow these password requirements:
- Minimum **14 characters** with a mix of uppercase, lowercase, numbers, and symbols
- Passwords must be changed every **90 days**
- Previous **12 passwords** cannot be reused
- Accounts are locked after **5 failed login attempts** (auto-unlock after 30 minutes)

### Password Reset
To reset your password:
1. Go to **https://identity.technova.com/reset**
2. Enter your employee email address
3. Complete the MFA verification
4. Set your new password following the complexity requirements
5. The new password takes effect immediately across all systems

For assistance, contact the IT Help Desk at **helpdesk@technova.com** or extension **4357**.

## Multi-Factor Authentication (MFA)

MFA is **mandatory** for all employees. TechNova supports:
- **Authenticator App** (recommended): Microsoft Authenticator or Google Authenticator
- **Hardware Security Key**: YubiKey (available from IT upon request)
- **SMS OTP**: Available as a backup method only

### MFA Setup
1. Log into the Identity Portal at **https://identity.technova.com**
2. Navigate to **Security → Multi-Factor Authentication**
3. Scan the QR code with your authenticator app
4. Enter the 6-digit verification code
5. Save the backup recovery codes in a secure location

## VPN Access

All remote access to company systems requires VPN connection:
- **Client**: Cisco AnyConnect (download from the IT portal)
- **Server**: vpn.technova.com
- **Authentication**: Employee credentials + MFA

### VPN Setup
1. Download Cisco AnyConnect from **https://it.technova.com/downloads**
2. Install and launch the application
3. Enter server address: **vpn.technova.com**
4. Log in with your employee credentials
5. Complete MFA when prompted

The VPN supports **split tunneling** — only corporate traffic is routed through the VPN.

## BYOD (Bring Your Own Device) Policy

Employees may use personal devices for work with the following conditions:
- Device must be enrolled in TechNova's Mobile Device Management (MDM) system
- Automatic screen lock must be enabled (maximum 2 minutes)
- Device encryption must be active
- Company data must be stored in the managed work profile only
- TechNova reserves the right to remotely wipe the **work profile** (not personal data) if the device is lost or the employee leaves
- Jailbroken or rooted devices are **not permitted**

## Data Classification

TechNova uses a **four-tier data classification** system:

| Level | Label | Description | Examples |
|-------|-------|-------------|----------|
| 1 | **Public** | Freely shareable | Marketing materials, blog posts |
| 2 | **Internal** | For employees only | Org charts, internal memos |
| 3 | **Confidential** | Business-sensitive | Financial projections, customer lists |
| 4 | **Restricted** | Highest protection | PII, source code, trade secrets |

### Handling Requirements
- **Public**: No special handling required
- **Internal**: Store on company systems, do not post externally
- **Confidential**: Encrypt at rest and in transit, access logged, need-to-know basis
- **Restricted**: End-to-end encryption, access requires VP approval, full audit trail

## Incident Response

### How to Report a Security Incident
1. **Immediately** email **security@technova.com** with details
2. Call the Security Operations Center (SOC) at extension **9111** for urgent incidents
3. Do **not** attempt to investigate or contain the incident yourself
4. Preserve all evidence (do not delete emails, files, or logs)

### Incident Categories
- **P1 (Critical)**: Active data breach, ransomware → Response within 15 minutes
- **P2 (High)**: Suspected compromise, phishing with credential exposure → Response within 1 hour
- **P3 (Medium)**: Policy violations, suspicious activity → Response within 4 hours
- **P4 (Low)**: General security questions, false positives → Response within 24 hours

## IT Support

The IT Help Desk is available:
- **Hours**: Monday–Friday, 8:00 AM – 8:00 PM (all time zones)
- **Email**: helpdesk@technova.com
- **Phone**: Extension 4357 (HELP)
- **Portal**: https://it.technova.com/support
- **Walk-in**: Building A, Floor 2, Room 201

For after-hours emergencies, call the 24/7 on-call number at **+1-555-TECH-911**.

## Software Requests

To request new software or tools:
1. Submit a request through the IT Service Portal
2. IT Security reviews the application for compliance
3. Procurement handles licensing (typically 5–10 business days)
4. IT deploys the software to your device

Pre-approved software (available for self-service installation):
- VS Code, IntelliJ IDEA, PyCharm
- Slack, Zoom, Microsoft Teams
- Chrome, Firefox
- Docker Desktop, Postman

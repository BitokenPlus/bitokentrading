# get_public_ip.py
import requests
import json

def get_public_ip_address():
    """
    Obtiene la dirección IP pública consultando varios servicios externos.
    Devuelve la IP como string o None si falla.
    """
    # Lista de servicios API para obtener la IP pública
    # Algunos pueden fallar o estar caídos, por eso probamos varios.
    ip_services = [
        "https://api.ipify.org?format=json",  # Devuelve JSON: {"ip":"1.2.3.4"}
        "https://httpbin.org/ip",            # Devuelve JSON: {"origin": "1.2.3.4, 5.6.7.8"} (puede tener proxies)
        "https://ipinfo.io/json",            # Devuelve JSON con más info, buscamos 'ip'
        "https://checkip.amazonaws.com",     # Devuelve solo la IP como texto plano
        "https://api.myip.com"               # Devuelve JSON
    ]

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    for service_url in ip_services:
        try:
            print(f"Intentando con: {service_url}")
            response = requests.get(service_url, headers=headers, timeout=5) # Timeout de 5 segundos
            response.raise_for_status() # Lanza una excepción para códigos de error HTTP (4xx o 5xx)

            if "ipify.org" in service_url or "myip.com" in service_url:
                # Estos devuelven JSON con una clave 'ip' simple
                ip_data = response.json()
                public_ip = ip_data.get('ip')
                if public_ip:
                    return public_ip
            elif "httpbin.org" in service_url:
                # httpbin.org devuelve {"origin": "ip1, ip2, ..."}
                # Tomamos la primera IP de la lista 'origin'
                ip_data = response.json()
                origin_ips = ip_data.get('origin')
                if origin_ips:
                    public_ip = origin_ips.split(',')[0].strip()
                    return public_ip
            elif "ipinfo.io" in service_url:
                ip_data = response.json()
                public_ip = ip_data.get('ip')
                if public_ip:
                    return public_ip
            elif "amazonaws.com" in service_url:
                # Este devuelve la IP como texto plano
                public_ip = response.text.strip()
                if public_ip: # Asegurarse que no esté vacío y sea una IP válida (básica)
                    if '.' in public_ip and len(public_ip.split('.')) == 4:
                         return public_ip
            
            print(f"Respuesta inesperada o sin IP de {service_url}: {response.text[:100]}")

        except requests.exceptions.RequestException as e:
            print(f"Error al conectar con {service_url}: {e}")
        except json.JSONDecodeError as e:
            print(f"Error al decodificar JSON de {service_url}: {e}. Respuesta: {response.text[:100]}")
        except Exception as e:
            print(f"Error inesperado con {service_url}: {e}")
            
    print("No se pudo obtener la IP pública de ninguno de los servicios.")
    return None

if __name__ == "__main__":
    my_ip = get_public_ip_address()
    if my_ip:
        print(f"\nTu dirección IP pública es: {my_ip}")
        print("Esta es la IP que debes agregar a la lista blanca de Binance si tienes restricciones de IP habilitadas.")
    else:
        print("\nNo se pudo determinar tu dirección IP pública.")